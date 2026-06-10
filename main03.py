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
# 5. EXECUTE OPTIMIZATION & EXTRACT DATA
# ==========================================
if __name__ == "__main__":
    # Generate a linear initial guess for the interior spline points
    cp_guess = np.linspace(q_start, q_end, 5)[1:-1].T.flatten()

    x0 = np.concatenate((
        [100.0, 0.5],             # k, L0
        [0.5, 0.5, 1.0],          # Ground Anchor X, Y, Z 
        [0.0, 0.0, 0.0],          # Robot Local Anchor
        cp_guess                  # Spline control points 
    ))

    bounds = [
        (0, 2000), (0.1, 2.0), 
        (-2, 2), (-2, 2), (0, 3), 
        (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1) 
    ]
    # Tight bounds: allow +/- 1.0 radian deviation from standard path
    for guess in cp_guess:
        bounds.append((guess - 1.0, guess + 1.0))

    # ---------------------------------------------------------
    # RUN OPTIMIZER
    # ---------------------------------------------------------
    print("Running Baseline Evaluation (Rigid Arm, Standard Spline)...")
    baseline_energy = objective_function(np.concatenate(([0.0, 0.0, 0,0,0, 0,0,0], cp_guess)))

    print("Starting NLP Optimization (SLSQP)...")
    import time
    start_time = time.time()
    
    result = minimize(
        objective_function, x0, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'disp': True}
    )

    # ---------------------------------------------------------
    # PRINT REQUESTED SETTINGS & PARAMETERS
    # ---------------------------------------------------------
    optimized_cp_interior = result.x[8:].reshape(6, 3)

    print(f"\n--- OPTIMIZATION COMPLETE ({time.time() - start_time:.1f}s) ---")
    print(f"Success Status: {result.success}")
    
    print("\n--- TASK SETTINGS ---")
    print(f"Point 1 (Start Angles): {np.round(q_start, 3)}")
    print(f"Point 2 (End Angles):   {np.round(q_end, 3)}")
    
    print("\n--- OPTIMIZED SPLINE PARAMETERS (Interior Knots) ---")
    joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
    for j in range(6):
        print(f"{joint_names[j]:<15}: {np.round(optimized_cp_interior[j], 3)}")

    print("\n--- ENERGY COMPARISON ---")
    print(f"Baseline (No Spring, Std Spline): {baseline_energy:.2f}")
    print(f"Optimized (With Spring):          {result.fun:.2f}")
    print(f"Energy Reduction:                 {((baseline_energy - result.fun)/baseline_energy)*100:.1f}%")
    
    print("\n--- OPTIMIZED HARDWARE ---")
    print(f"Stiffness (k): {result.x[0]:.2f} N/m")
    print(f"Rest Length (L0): {result.x[1]:.2f} m")

    # ==========================================
    # 6. CALCULATE ENERGIES FOR PLOTTING
    # ==========================================
    print("\nGenerating physics data for plots...")
    
    # Extract optimized hardware parameters
    k_opt, L0_opt = result.x[0], result.x[1]
    P_g_opt = result.x[2:5]
    P_r_local_opt = result.x[5:8]
    
    # Generate the actual kinematic arrays
    q_opt, v_opt, a_opt = generate_trajectory(optimized_cp_interior)
    t_eval = np.linspace(0, T_total, N_steps)
    
    # Arrays to hold physics data over time
    KE_arr = np.zeros(N_steps)
    PE_robot_arr = np.zeros(N_steps)
    PE_spring_arr = np.zeros(N_steps)
    
    for i in range(N_steps):
        q, v = q_opt[i], v_opt[i]
        
        # Calculate Robot Kinetic Energy (0.5 * v^T * M * v)
        KE_arr[i] = pin.computeKineticEnergy(model, data, q, v)
        
        # Calculate Robot Potential Energy (Gravity)
        PE_robot_arr[i] = pin.computePotentialEnergy(model, data, q)
        
        # Calculate Spring Potential Energy (0.5 * k * dx^2)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        P_r_global = data.oMf[frame_id].act(P_r_local_opt)
        
        L = np.linalg.norm(P_g_opt - P_r_global)
        if L > L0_opt:
            PE_spring_arr[i] = 0.5 * k_opt * (L - L0_opt)**2

    # ==========================================
    # 7. VISUALIZATION
    # ==========================================
    import matplotlib.pyplot as plt
    
    # --- FIGURE 1: TRAJECTORY COMPARISON ---
    fig1 = plt.figure(figsize=(14, 8))
    fig1.canvas.manager.set_window_title('Trajectory Comparison')
    plt.suptitle("Optimized vs. Standard Clamped Spline", fontsize=16)
    
    # Generate a pure standard clamped spline for comparison
    q_std, _, _ = generate_trajectory(cp_guess.reshape(6,3))
    
    for j in range(6):
        plt.subplot(2, 3, j+1)
        plt.plot(t_eval, q_opt[:, j], label='Optimized (Spring)', color='red', linewidth=2)
        plt.plot(t_eval, q_std[:, j], label='Baseline (No Spring)', linestyle='--', color='blue')
        plt.title(joint_names[j])
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.grid(True, linestyle=':', alpha=0.7)
        if j == 0: plt.legend()
    plt.tight_layout()

    # --- FIGURE 2: SYSTEM ENERGY DYNAMICS ---
    fig2 = plt.figure(figsize=(10, 6))
    fig2.canvas.manager.set_window_title('Energy Dynamics')
    plt.suptitle("System Energy over Time (Virtual Pendulum Effect)", fontsize=16)
    
    plt.plot(t_eval, KE_arr, label='Kinetic Energy (Robot)', color='orange', linewidth=2)
    plt.plot(t_eval, PE_robot_arr - PE_robot_arr[0], label='Gravitational PE (Robot, Relative)', color='blue', linewidth=2)
    plt.plot(t_eval, PE_spring_arr, label='Elastic PE (Spring)', color='green', linewidth=2)
    
    # Total System Mechanical Energy
    Total_E = KE_arr + (PE_robot_arr - PE_robot_arr[0]) + PE_spring_arr
    plt.plot(t_eval, Total_E, label='Total Mechanical Energy', color='black', linestyle='--', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.show()