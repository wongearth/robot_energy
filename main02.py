import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
from scipy.interpolate import CubicSpline
import pinocchio as pin
import time
import matplotlib.pyplot as plt

# ==========================================
# 1. BUILD EXACT UR5 MODEL IN PYTHON MEMORY
# ==========================================
def build_ur5_manually():
    """
    Bypasses broken URDF parsers by mathematically constructing 
    the UR5 using exact masses, COM offsets, and kinematic placements.
    """
    model = pin.Model()
    
    # Exact UR5 Link Masses (kg)
    masses = [3.7, 8.393, 2.275, 1.219, 1.219, 0.1879]
    
    # Exact UR5 Kinematics (SE3 Placements relative to previous joint)
    placements = [
        pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.089159])),                           # Shoulder Pan
        pin.SE3(pin.utils.rpyToMatrix(np.pi/2, 0, 0), np.array([0.0, 0.0, 0.0])),     # Shoulder Lift
        pin.SE3(np.eye(3), np.array([-0.425, 0.0, 0.0])),                             # Elbow
        pin.SE3(np.eye(3), np.array([-0.39225, 0.0, 0.0])),                           # Wrist 1
        pin.SE3(pin.utils.rpyToMatrix(np.pi/2, 0, 0), np.array([0.0, 0.10915, 0.0])), # Wrist 2
        pin.SE3(pin.utils.rpyToMatrix(-np.pi/2, 0, 0), np.array([0.0, 0.0, 0.09465])) # Wrist 3
    ]
    
    # Accurate Center of Mass offsets relative to each joint frame
    com_offsets = [
        np.array([0.0, 0.0, 0.0]),             # Shoulder Pan
        np.array([-0.2125, 0.0, 0.0]),         # Shoulder Lift (Upper Arm)
        np.array([-0.196125, 0.0, 0.0]),       # Elbow (Forearm)
        np.array([0.0, 0.0, 0.0]),             # Wrist 1
        np.array([0.0, 0.0, 0.0]),             # Wrist 2
        np.array([0.0, 0.0, 0.0])              # Wrist 3
    ]
    
    parent_id = 0
    for i in range(6):
        # Add revolute joint around Z axis
        joint_id = model.addJoint(parent_id, pin.JointModelRZ(), placements[i], f"joint_{i+1}")
        
        # Build the exact placement for the COM
        com_placement = pin.SE3(np.eye(3), com_offsets[i])
        
        # Add inertial properties (Using approximated spherical inertia for solver speed)
        inertia = pin.Inertia.FromSphere(masses[i], 0.05)
        model.appendBodyToJoint(joint_id, inertia, com_placement)
        
        parent_id = joint_id
        
    # Create an operational frame for the spring attachment point (attached to Wrist 2)
    # You can change parent_id to attach it to a different link
    model.addFrame(pin.Frame("spring_attach", parent_id, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    
    return model

# Initialize the mathematical model globally
model = build_ur5_manually()
data = model.createData()
frame_id = model.getFrameId("spring_attach")

# ==========================================
# 2. TASK PARAMETERS
# ==========================================
T_total = 2.0  # Total time for the point-to-point motion
N_steps = 30   # Discrete integration steps (keep at 30-50 for faster NLP iterations)
dt = T_total / N_steps

# Start and End joint configurations (Radians)
q_start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
q_end = np.array([1.57, -1.0, 1.0, -1.0, 1.57, 0.0])

# ==========================================
# 3. TRAJECTORY GENERATOR (CLAMPED)
# ==========================================
def generate_trajectory(control_points_interior):
    """Generates continuous q, q_dot, q_ddot arrays with ZERO start/end velocity."""
    t_eval = np.linspace(0, T_total, N_steps)
    t_knots = np.linspace(0, T_total, 5) 
    
    q_traj = np.zeros((N_steps, 6))
    v_traj = np.zeros((N_steps, 6))
    a_traj = np.zeros((N_steps, 6))
    
    for j in range(6):
        # 1. Combine fixed start, the 3 optimized interior points, and fixed end
        cp = np.concatenate(([q_start[j]], control_points_interior[j], [q_end[j]]))
        
        # 2. Create the spline. 
        # bc_type='clamped' forces the first derivative (velocity) to be ZERO at both ends!
        spline = CubicSpline(t_knots, cp, bc_type='clamped')
        
        # 3. Evaluate the spline and its derivatives
        q_traj[:, j] = spline(t_eval)
        v_traj[:, j] = spline(t_eval, 1) # First derivative (Velocity)
        a_traj[:, j] = spline(t_eval, 2) # Second derivative (Acceleration)
        
    return q_traj, v_traj, a_traj

# ==========================================
# 4. OBJECTIVE FUNCTION (WITH REGULARIZATION)
# ==========================================
def objective_function(x):
    # Unpack variables
    k, L0 = x[0], x[1]
    P_g = x[2:5]
    P_r_local = x[5:8] 
    cp_interior = x[8:].reshape(6, 3)
    
    q_traj, v_traj, a_traj = generate_trajectory(cp_interior)
    
    total_energy = 0.0
    path_penalty = 0.0
    velocity_penalty = 0.0
    
    for i in range(N_steps):
        q, v, a = q_traj[i], v_traj[i], a_traj[i]
        
        # 1. Rigid body dynamics
        tau_rigid = pin.rnea(model, data, q, v, a)
        
        # 2. Kinematics for spring attachment point
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        P_r_global = data.oMf[frame_id].act(P_r_local)
        
        # 3. Calculate 3D Spring Force Vector
        dist_vec = P_g - P_r_global
        L = np.linalg.norm(dist_vec)
        
        F_spring_3D = np.zeros(3)
        if L > L0: 
            F_spring_3D = (k * (L - L0)) * (dist_vec / L)
            
        # 4. Map Cartesian Force to Joint Torques via the Jacobian
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        tau_spring = J[:3, :].T @ F_spring_3D
        
        # 5. Integrate squared net motor torque (The true objective)
        tau_motor = tau_rigid - tau_spring
        total_energy += np.sum(tau_motor**2) * dt
        
        # ----------------------------------------------------
        # THE PENALTY BOX (Stops the optimizer from cheating)
        # ----------------------------------------------------
        # Calculate where the robot *should* be on a normal straight path at this time step
        q_ideal = q_start + (q_end - q_start) * (i / (N_steps - 1))
        
        # Penalize deviating from the straight path (Weight = 500)
        path_penalty += np.sum((q - q_ideal)**2) * 500.0 * dt
        
        # Penalize insanely high velocities (Weight = 50)
        velocity_penalty += np.sum(v**2) * 50.0 * dt

    # The solver tries to minimize the sum of all three
    return total_energy + path_penalty + velocity_penalty

# ==========================================
# 5. EXECUTE OPTIMIZATION
# ==========================================
if __name__ == "__main__":
    # Generate a linear initial guess for the interior spline points
    cp_guess = np.linspace(q_start, q_end, 5)[1:-1].T.flatten()

    # Pack initial guess x0 (26 variables total)
    x0 = np.concatenate((
        [100.0, 0.5],             # k (N/m), L0 (m)
        [0.5, 0.5, 1.0],          # Ground Anchor X, Y, Z (m)
        [0.0, 0.0, 0.0],          # Robot Local Anchor X, Y, Z (m)
        cp_guess                  # Spline control points (18 variables)
    ))

    # Define physical bounds
    bounds = [
        (0, 2000),      # Stiffness k bound (N/m)
        (0.1, 2.0),     # Rest length L0 bound (m)
        (-2, 2), (-2, 2), (0, 3), # Ground anchor X, Y, Z bounds
        (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1) # Robot anchor must stay near the link center
    ]
    # Add loose bounds for the 18 joint angle control points (-2pi to 2pi radians)
    bounds += [(-6.28, 6.28)] * 18 

    # # NEW TIGHTER BOUNDS: Force the optimizer to stay within +/- 1.0 radian 
    # # of the initial linear guess, so it doesn't swing the arm through the floor.
    # for guess in cp_guess:
    #     bounds.append((guess - 1.0, guess + 1.0))

    print("Running Baseline Evaluation (Rigid Arm Only)...")
    baseline_energy = objective_function(np.concatenate(([0.0, 0.0, 0,0,0, 0,0,0], cp_guess)))
    print(f"Baseline Squared Torque Integral: {baseline_energy:.2f}\n")

    print("Starting NLP Optimization (SLSQP). This may take a minute...")
    start_time = time.time()
    
    result = minimize(
        objective_function, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        options={'maxiter': 1000, 'disp': True}
    )

    print(f"\n--- OPTIMIZATION COMPLETE ({time.time() - start_time:.1f}s) ---")
    print(f"Success Status: {result.success}")
    print(f"Original Energy:  {baseline_energy:.2f}")
    print(f"Optimized Energy: {result.fun:.2f}")
    print(f"Energy Reduction: {((baseline_energy - result.fun)/baseline_energy)*100:.1f}%\n")
    
    print("--- OPTIMIZED HARDWARE PARAMETERS ---")
    print(f"Stiffness (k): {result.x[0]:.2f} N/m")
    print(f"Rest Length (L0): {result.x[1]:.2f} m")
    print(f"Ground Anchor (Global X,Y,Z): [{result.x[2]:.3f}, {result.x[3]:.3f}, {result.x[4]:.3f}]")
    print(f"Robot Anchor (Local X,Y,Z): [{result.x[5]:.3f}, {result.x[6]:.3f}, {result.x[7]:.3f}]")

    # ==========================================
    # 6. VISUALIZE THE "CHEATING" TRAJECTORY
    # ==========================================
    print("Generating trajectory plots...")
    
    # Extract the optimized control points from the result
    optimized_cp_interior = result.x[8:].reshape(6, 3)
    
    # Generate the actual time-series arrays for the optimized path
    q_opt, v_opt, a_opt = generate_trajectory(optimized_cp_interior)
    
    # Time array for the X-axis
    t_eval = np.linspace(0, T_total, N_steps)
    
    # Setup the plot
    plt.figure(figsize=(12, 8))
    joint_names = ["Shoulder Pan", "Shoulder Lift (Upper Arm)", "Elbow (Forearm)", 
                   "Wrist 1", "Wrist 2", "Wrist 3"]
    
    plt.suptitle("Optimized vs. Standard Linear Trajectory", fontsize=16)
    
    for j in range(6):
        plt.subplot(2, 3, j+1)
        
        # Plot the optimizer's crazy path
        plt.plot(t_eval, q_opt[:, j], label='Optimized Path', color='red', linewidth=2)
        
        # Plot a standard, rigid point-to-point path for comparison
        q_linear = np.linspace(q_start[j], q_end[j], N_steps)
        plt.plot(t_eval, q_linear, label='Standard Path', linestyle='--', color='blue')
        
        plt.title(joint_names[j])
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (Radians)')
        plt.grid(True, linestyle=':', alpha=0.7)
        if j == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.show()