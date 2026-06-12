import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import pinocchio as pin
import example_robot_data as erd

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
# Load the official UR5 model
robot = erd.load('ur5')
model = robot.model
data = robot.data

# Choose the link to attach the robot side of the spring (e.g., forearm or wrist)
# We will use 'wrist_2_link' as the distal attachment point
frame_name = "wrist_2_link"
frame_id = model.getFrameId(frame_name)

# Task Parameters
T_total = 2.0      # Total time for the motion (seconds)
N_steps = 50       # Number of discrete integration steps
dt = T_total / N_steps

# Define starting and ending joint configurations (q1 and q2)
q_start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
q_end = np.array([1.57, -1.0, 1.0, -1.0, 1.57, 0.0])

# ==========================================
# 2. TRAJECTORY GENERATOR
# ==========================================
def generate_trajectory(control_points_interior):
    """
    Uses B-Splines to generate q(t), q_dot(t), q_ddot(t) 
    based on the NLP's control point guesses.
    """
    t_eval = np.linspace(0, T_total, N_steps)
    t_knots = np.linspace(0, T_total, 5) # 5 points: Start, CP1, CP2, CP3, End
    
    q_traj = np.zeros((N_steps, 6))
    v_traj = np.zeros((N_steps, 6))
    a_traj = np.zeros((N_steps, 6))
    
    for j in range(6):
        # Combine fixed start, 3 optimized interior points, and fixed end
        cp = np.concatenate(([q_start[j]], control_points_interior[j], [q_end[j]]))
        
        # Create a cubic B-spline
        spline = make_interp_spline(t_knots, cp, k=3)
        
        q_traj[:, j] = spline(t_eval)
        v_traj[:, j] = spline(t_eval, nu=1) # First derivative (Velocity)
        a_traj[:, j] = spline(t_eval, nu=2) # Second derivative (Acceleration)
        
    return q_traj, v_traj, a_traj

# ==========================================
# 3. THE OBJECTIVE FUNCTION
# ==========================================
def objective_function(x):
    """
    Evaluates the total energy cost for a given guess of the 26 variables.
    """
    # 1. Unpack Variables [cite: 195]
    k, L0 = x[0], x[1]
    P_g = x[2:5]         # Ground Anchor X, Y, Z
    P_r_local = x[5:8]   # Robot Anchor X, Y, Z (Local to link)
    cp_interior = x[8:].reshape(6, 3) # 18 Trajectory Control Points [cite: 192, 193]
    
    # 2. Generate the Path [cite: 196]
    q_traj, v_traj, a_traj = generate_trajectory(cp_interior)
    
    total_energy = 0.0
    
    # Loop over time steps
    for i in range(N_steps):
        q = q_traj[i]
        v = v_traj[i]
        a = a_traj[i]
        
        # 3. Run Rigid Dynamics [cite: 197]
        # tau_rigid = M*a + C*v + G
        tau_rigid = pin.rnea(model, data, q, v, a)
        
        # 4. Apply Spring Logic [cite: 198]
        # Forward kinematics to find the robot attachment point in global space
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Convert local offset to global coordinates [cite: 212, 213]
        P_r_global = data.oMf[frame_id].act(P_r_local)
        
        # Calculate distance L [cite: 124, 162]
        dist_vec = P_g - P_r_global
        L = np.linalg.norm(dist_vec)
        
        # Calculate 3D Spring Force Vector
        F_spring_3D = np.zeros(3)
        if L > L0: # Spring force is max(0, k*(L-L0)) [cite: 115, 157]
            force_mag = k * (L - L0)
            F_spring_3D = force_mag * (dist_vec / L) # Normalized direction [cite: 126, 158]
            
        # Map Cartesian Force to Joint Torques via the Jacobian
        # MUST use LOCAL_WORLD_ALIGNED to match the 3D global force vector [cite: 214, 215]
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # tau_spring = J^T * F_spring [cite: 93, 159]
        tau_spring = J[:3, :].T @ F_spring_3D 
        
        # Net required motor torque [cite: 164]
        tau_motor = tau_rigid - tau_spring
        
        # 5. Calculate Cost [cite: 199]
        # Integrate sum of squared motor torques [cite: 116, 165]
        total_energy += np.sum(tau_motor**2) * dt
        
    return total_energy

# ==========================================
# 4. EXECUTE OPTIMIZATION
# ==========================================
if __name__ == "__main__":
    
    # Initial Guess (x0)
    # Generate a simple linear interpolation for the interior spline points
    cp_guess = np.linspace(q_start, q_end, 5)[1:-1].T.flatten()
    
    x0 = np.concatenate((
        [500.0, 0.5],             # k (N/m), L0 (m)
        [0.5, 0.5, 1.0],          # Ground Anchor X, Y, Z (m)
        [0.0, 0.0, 0.0],          # Robot Local Anchor X, Y, Z (m)
        cp_guess                  # Spline control points
    ))
    
    # Define bounds to prevent physically impossible solutions [cite: 168, 169, 170]
    bounds = [
        (0, 5000),      # k bound
        (0.1, 2.0),     # L0 bound
        (-2, 2), (-2, 2), (0, 3), # Ground anchor limits (Room box)
        (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1) # Robot anchor limits (Link box)
    ]
    # Add bounds for the 18 joint angle control points (-2pi to 2pi radians) [cite: 200]
    bounds += [(-6.28, 6.28)] * 18 
    
    print("Running NLP Optimization (This may take a few minutes)...")
    
    # Run the SLSQP solver [cite: 153]
    result = minimize(
        objective_function, 
        x0, 
        method='SLSQP', 
        bounds=bounds,
        options={'maxiter': 200, 'disp': True}
    )
    
    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Success: {result.success}")
    print(f"Minimum Energy Cost: {result.fun:.2f}")
    print(f"Optimized Stiffness (k): {result.x[0]:.2f} N/m")
    print(f"Optimized Rest Length (L0): {result.x[1]:.2f} m")
    print(f"Ground Anchor (Global): {np.round(result.x[2:5], 3)}")
    print(f"Robot Anchor (Local): {np.round(result.x[5:8], 3)}")