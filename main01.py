import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline, make_interp_spline
import pinocchio as pin
import example_robot_data as erd

# ==========================================
# 1. LOAD ACCURATE UR5 KINEMATICS & DYNAMICS
# ==========================================
# This automatically loads the official UR5 URDF with accurate mass/inertia data
robot = erd.load('ur5')
model = robot.model
data = robot.data


# local
# urdf_path = "ur5.urdf" 

# # Build the model and data directly from the local file
# model = pin.buildModelFromUrdf(urdf_path)
# data = model.createData()


# Define the frame we are attaching the spring to (e.g., the wrist)
# You can check model.frames to find the exact string name
LINK_NAME = 'wrist_2_link' 
frame_id = model.getFrameId(LINK_NAME)

# ==========================================
# 2. TASK PARAMETERS (Fixed)
# ==========================================
T_total = 2.0  # Total time for the motion in seconds
N_steps = 50   # Number of discrete time steps for integration
dt = T_total / N_steps

# Start and End joint configurations (Radians)
q_start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
q_end = np.array([1.57, -1.0, 1.0, -1.0, 1.57, 0.0])

# ==========================================
# 3. TRAJECTORY GENERATOR (B-SPLINES)
# ==========================================
def generate_trajectory(control_points_interior, q_start, q_end, T_total, N_steps):
    """
    Generates q, q_dot, q_ddot arrays using B-Splines.
    control_points_interior is a (6, 3) array (3 interior points per joint)
    """
    t_eval = np.linspace(0, T_total, N_steps)
    t_knots = np.linspace(0, T_total, 5) # 5 knots for 3 interior points + start/end
    
    q_traj = np.zeros((N_steps, 6))
    v_traj = np.zeros((N_steps, 6))
    a_traj = np.zeros((N_steps, 6))
    
    for j in range(6):
        # Full control points for joint j: [start, interior_1, interior_2, interior_3, end]
        cp = np.concatenate(([q_start[j]], control_points_interior[j], [q_end[j]]))
        
        # Create spline (degree 3 for smooth acceleration)
        spline = make_interp_spline(t_knots, cp, k=3)
        spline_v = spline.derivative(nu=1)
        spline_a = spline.derivative(nu=2)
        
        q_traj[:, j] = spline(t_eval)
        v_traj[:, j] = spline_v(t_eval)
        a_traj[:, j] = spline_a(t_eval)
        
    return q_traj, v_traj, a_traj

# ==========================================
# 4. THE COST FUNCTION (ENERGY)
# ==========================================
def objective_function(x):
    """
    x contains 26 variables:
    x[0]: Spring stiffness (k)
    x[1]: Spring resting length (L0)
    x[2:5]: Ground anchor coordinates (Xg, Yg, Zg)
    x[5:8]: Robot anchor local offset (Xr, Yr, Zr) relative to the chosen link
    x[8:26]: 18 Spline control points (3 per joint for 6 joints)
    """
    # Unpack variables
    k = x[0]
    L0 = x[1]
    P_g = x[2:5]
    P_r_local = x[5:8] 
    
    # Reshape control points back to (6, 3) matrix
    cp_interior = x[8:].reshape(6, 3)
    
    # Generate kinematic trajectory
    q_traj, v_traj, a_traj = generate_trajectory(cp_interior, q_start, q_end, T_total, N_steps)
    
    total_energy = 0.0
    
    for i in range(N_steps):
        q = q_traj[i]
        v = v_traj[i]
        a = a_traj[i]
        
        # 1. Calculate rigid body torques using Pinocchio (RNEA)
        tau_rigid = pin.rnea(model, data, q, v, a)
        
        # 2. Forward Kinematics to find 3D position of the robot anchor
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get frame placement and compute exact global position of the local anchor
        frame_placement = data.oMf[frame_id]
        P_r_global = frame_placement.act(P_r_local)
        
        # 3. Calculate Spring Force
        distance_vector = P_g - P_r_global
        L = np.linalg.norm(distance_vector)
        
        if L > L0:
            force_magnitude = k * (L - L0)
            force_direction = distance_vector / L
            F_spring_3D = force_magnitude * force_direction
        else:
            F_spring_3D = np.zeros(3)
            
        # 4. Calculate Jacobian to map 3D force to joint torques
        # We need the translational part of the Jacobian in the world frame
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_trans = J[:3, :] # Extract top 3 rows (X, Y, Z translation)
        
        # tau_spring = J^T * F
        tau_spring = J_trans.T @ F_spring_3D
        
        # 5. Net required motor torque
        tau_motor = tau_rigid - tau_spring
        
        # 6. Accumulate cost (Integral of sum of squared torques)
        total_energy += np.sum(tau_motor**2) * dt
        
    return total_energy

# ==========================================
# 5. NLP OPTIMIZATION SETUP & EXECUTION
# ==========================================
# Initial Guesses
k_guess = 500.0
L0_guess = 0.5
Pg_guess = [0.5, 0.5, 1.0] # Ground anchor
Pr_guess = [0.0, 0.0, 0.0] # Center of the wrist link
cp_guess = np.linspace(q_start, q_end, 5)[1:-1].T.flatten() # Linear interpolation for initial spline points

x0 = np.concatenate(([k_guess, L0_guess], Pg_guess, Pr_guess, cp_guess))

# Bounds to prevent physically impossible scenarios
bounds = [
    (0, 5000),      # k: 0 to 5000 N/m
    (0.1, 2.0),     # L0: 0.1m to 2.0m
    (-2, 2), (-2, 2), (0, 3), # Ground anchor X, Y, Z bounds
    (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1) # Robot anchor must stay near the link center
]
# Add loose bounds for the 18 joint angle control points (-2pi to 2pi)
bounds += [(-6.28, 6.28)] * 18 

print("Starting NLP Optimization. This may take a few minutes...")

# Run the SLSQP solver
result = minimize(
    objective_function, 
    x0, 
    method='SLSQP', 
    bounds=bounds,
    options={'maxiter': 1000, 'disp': True}
)

print("\n--- OPTIMIZATION RESULTS ---")
print(f"Success: {result.success}")
print(f"Optimized Energy Cost: {result.fun:.2f}")
print(f"Optimized Stiffness (k): {result.x[0]:.2f} N/m")
print(f"Optimized Rest Length (L0): {result.x[1]:.2f} m")
print(f"Ground Anchor (X,Y,Z): {result.x[2:5]}")