import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import pinocchio as pin
import time
import matplotlib.pyplot as plt

# ==========================================
# 1. BUILD EXACT UR5 MODEL IN PYTHON MEMORY
# ==========================================

# Initialize the mathematical model globally
urdf_path = "/home/wong/ws_robot_energy/ur5.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Dynamically find the correct integer ID for the spring attachment point
attachment_link_name = "wrist_2_link" 
if model.existFrame(attachment_link_name):
    frame_id = model.getFrameId(attachment_link_name)
    print(f"Spring attached to '{attachment_link_name}' at Frame ID: {frame_id}")
else:
    raise ValueError(f"Link '{attachment_link_name}' does not exist in the URDF!")

# Dynamically find the end-effector frame for the 3D plot
if model.existFrame("tool0"):
    tool_frame_id = model.getFrameId("tool0")
else:
    tool_frame_id = model.getFrameId("wrist_3_link")

# ==========================================
# 2. TASK PARAMETERS
# ==========================================
T_total = 1  # Total time for the point-to-point motion
N_steps = 30   # Discrete integration steps (keep at 30-50 for faster NLP iterations)
dt = T_total / N_steps

# Point 1 (Pick): Floor-level
q_start = np.array([0.0, -1.0, 2.0, -2.57, -1.57, 0.0])

# Point MID (Clearance): Halfway through the time. 
# Base at 45 deg (0.785 rad). Arm pulled UP to clear the obstacle.
q_mid = np.array([0.785, -1.57, 0.5, -0.5, -1.57, 0.0])

# Point 2 (Place): Floor-level
q_end = np.array([1.57, -0.8, 1.8, -2.57, -1.57, 0.0])

# ==========================================
# 3. TRAJECTORY GENERATOR (7-Knot Via-Point)
# ==========================================
def generate_trajectory(control_points_interior):
    """Generates continuous arrays passing through Start -> Mid -> End."""
    t_eval = np.linspace(0, T_total, N_steps)
    
    # 7 Time knots evenly spaced: 0, 0.33, 0.66, 1.0 (Mid), 1.33, 1.66, 2.0
    t_knots = np.linspace(0, T_total, 7) 
    
    q_traj = np.zeros((N_steps, 6))
    v_traj = np.zeros((N_steps, 6))
    a_traj = np.zeros((N_steps, 6))
    
    for j in range(6):
        # Unpack the 4 optimized interior points for this joint
        cp_j = control_points_interior[j] 
        
        # Build the 7-point path: Start -> cp0 -> cp1 -> MID -> cp2 -> cp3 -> End
        full_path = np.array([
            q_start[j], 
            cp_j[0], cp_j[1], 
            q_mid[j], 
            cp_j[2], cp_j[3], 
            q_end[j]
        ])
        
        # Create spline (Still clamped at the very ends, but smooth at the Mid point!)
        spline = CubicSpline(t_knots, full_path, bc_type='clamped')
        
        q_traj[:, j] = spline(t_eval)
        v_traj[:, j] = spline(t_eval, 1) # Velocity
        a_traj[:, j] = spline(t_eval, 2) # Acceleration
        
    return q_traj, v_traj, a_traj

# ==========================================
# 4. OBJECTIVE FUNCTION (CYCLIC ROUND TRIP)
# ==========================================
def objective_function(x):
    k, L0 = x[0], x[1]
    P_g = x[2:5]
    P_r_local = x[5:8] 
    cp_interior = x[8:].reshape(6, 4) # New 4-point structure!    
    
    # 1. Generate the forward path (Point 1 -> Point 2)
    q_fwd, v_fwd, a_fwd = generate_trajectory(cp_interior)
    
    # 2. Mathematically mirror the path for the return trip (Point 2 -> Point 1)
    q_ret = q_fwd[::-1]
    v_ret = -v_fwd[::-1]  # Velocity reverses direction!
    a_ret = a_fwd[::-1]   # Acceleration remains symmetric
    
    # 3. Concatenate into a full repetitive cycle
    q_cycle = np.concatenate((q_fwd, q_ret), axis=0)
    v_cycle = np.concatenate((v_fwd, v_ret), axis=0)
    a_cycle = np.concatenate((a_fwd, a_ret), axis=0)
    
    total_energy = 0.0
    
    # Loop over the FULL cycle (2 * N_steps)
    for i in range(2 * N_steps):
        q, v, a = q_cycle[i], v_cycle[i], a_cycle[i]
        
        # Rigid body dynamics
        tau_rigid = pin.rnea(model, data, q, v, a)
        
        # CRITICAL FIX: Computes forward kinematics for all joints and operational frames
        pin.framesForwardKinematics(model, data, q)
        
        # Calculate 3D Spring Force Vector
        P_r_global = data.oMf[frame_id].act(P_r_local)
        dist_vec = P_g - P_r_global
        L = np.linalg.norm(dist_vec)
        
        F_spring_3D = np.zeros(3)
        if L > L0: 
            F_spring_3D = (k * (L - L0)) * (dist_vec / L)
            
        # Map Cartesian Force to Joint Torques via the Jacobian
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        tau_spring = J[:3, :].T @ F_spring_3D
        
        # Integrate squared net motor torque
        tau_motor = tau_rigid - tau_spring
        total_energy += np.sum(tau_motor**2) * dt
        
    # ----------------------------------------------------
    # THE PENALTY BOX (Calculated only on forward path)
    # ----------------------------------------------------
    path_penalty = 0.0
    velocity_penalty = 0.0
    
    for i in range(N_steps):
        q_ideal = q_start + (q_end - q_start) * (i / (N_steps - 1))
        path_penalty += np.sum((q_fwd[i] - q_ideal)**2) * 500.0 * dt
        velocity_penalty += np.sum(v_fwd[i]**2) * 50.0 * dt

    return total_energy + path_penalty + velocity_penalty

# ==========================================
# 5. EXECUTE OPTIMIZATION & EXTRACT DATA
# ==========================================
if __name__ == "__main__":
    # 1. Generate an initial linear guess that passes through the Mid point
    cp_guess = np.zeros((6, 4))
    for j in range(6):
        # 2 points between Start and Mid
        cp_guess[j, 0:2] = np.linspace(q_start[j], q_mid[j], 4)[1:-1]
        # 2 points between Mid and End
        cp_guess[j, 2:4] = np.linspace(q_mid[j], q_end[j], 4)[1:-1]
    
    cp_guess_flat = cp_guess.flatten() # Now 24 path variables

    # 2. Pack initial guess x0 (8 hardware vars + 24 path vars = 32 variables)
    x0 = np.concatenate((
        [100.0, 0.5],             # k, L0
        [0.5, 0.5, 1.0],          # Ground Anchor
        [0.0, 0.0, 0.0],          # Robot Local Anchor
        cp_guess_flat             # Spline control points 
    ))

    # 3. Define physical bounds
    bounds = [
        (0, 2000), (0.1, 2.0), 
        (-2, 2), (-2, 2), (0, 3), 
        (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1) 
    ]
    # Tight path bounds: allow +/- 1.0 radian deviation from the guess
    for guess in cp_guess_flat:
        bounds.append((guess - 1.0, guess + 1.0))

    # ---------------------------------------------------------
    # RUN OPTIMIZER
    # ---------------------------------------------------------
    print("Running Baseline Evaluation (Rigid Arm, Standard Spline)...")
    baseline_energy = objective_function(np.concatenate(([0.0, 0.0, 0,0,0, 0,0,0], cp_guess_flat)))
    print("Starting NLP Optimization (SLSQP)...")
    import time
    start_time = time.time()
    
    result = minimize(
        objective_function, x0, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'disp': True}
    )

    # ---------------------------------------------------------
    # PRINT REQUESTED SETTINGS & PARAMETERS
    # ---------------------------------------------------------
    optimized_cp_interior = result.x[8:].reshape(6, 4)

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
    # 6. CALCULATE ENERGIES FOR FULL CYCLE
    # ==========================================
    print("\nGenerating physics data for full cycle plots...")
    
    k_opt, L0_opt = result.x[0], result.x[1]
    P_g_opt = result.x[2:5]
    P_r_local_opt = result.x[5:8]
    
    q_fwd, v_fwd, a_fwd = generate_trajectory(optimized_cp_interior)
    q_opt = np.concatenate((q_fwd, q_fwd[::-1]), axis=0)
    v_opt = np.concatenate((v_fwd, -v_fwd[::-1]), axis=0)
    
    # Time array now spans 2x the total time
    t_eval = np.linspace(0, 2 * T_total, 2 * N_steps)
    
    KE_arr = np.zeros(2 * N_steps)
    PE_robot_arr = np.zeros(2 * N_steps)
    PE_spring_arr = np.zeros(2 * N_steps)
    
    for i in range(2 * N_steps):
        q, v = q_opt[i], v_opt[i]
        
        KE_arr[i] = pin.computeKineticEnergy(model, data, q, v)
        PE_robot_arr[i] = pin.computePotentialEnergy(model, data, q)
        
        # CRITICAL FIX: framesForwardKinematics computes the operational frames
        pin.framesForwardKinematics(model, data, q)
        P_r_global = data.oMf[frame_id].act(P_r_local_opt)
        
        L = np.linalg.norm(P_g_opt - P_r_global)
        if L > L0_opt:
            PE_spring_arr[i] = 0.5 * k_opt * (L - L0_opt)**2

    # ==========================================
    # 7. VISUALIZATION
    # ==========================================
    import matplotlib.pyplot as plt
    
    # --- FIGURE 1: CYCLIC TRAJECTORY (JOINT ANGLES VS TIME) ---
    fig1 = plt.figure(figsize=(14, 8))
    fig1.canvas.manager.set_window_title('Cyclic Trajectory Comparison')
    plt.suptitle("Optimized Joint Trajectories (Full Back-and-Forth Cycle)", fontsize=16)
    
    # Generate the standard clamped spline for the forward trip, then mirror it for a baseline
    q_std_fwd, _, _ = generate_trajectory(cp_guess.reshape(6,4))
    q_std_cycle = np.concatenate((q_std_fwd, q_std_fwd[::-1]), axis=0)
    
    for j in range(6):
        plt.subplot(2, 3, j+1)
        
        # Plot the Optimized Cyclic Path (Red)
        plt.plot(t_eval, q_opt[:, j], label='Optimized (Spring)', color='red', linewidth=2)
        
        # Plot the Standard Cyclic Path (Blue Dashed)
        plt.plot(t_eval, q_std_cycle[:, j], label='Baseline (No Spring)', linestyle='--', color='blue')
        
        # Add a vertical line marking the turnaround point
        plt.axvline(x=T_total, color='black', linestyle=':', alpha=0.5, label='Turnaround' if j==0 else "")
        
        plt.title(joint_names[j])
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.grid(True, linestyle=':', alpha=0.7)
        if j == 0: 
            plt.legend(loc='best')
            
    plt.tight_layout()
    
    # --- FIGURE 2: SYSTEM ENERGY DYNAMICS ---
    fig2 = plt.figure(figsize=(12, 7))
    fig2.canvas.manager.set_window_title('Energy Dynamics (Cyclic Task)')
    plt.suptitle("System Energy over Full Back-and-Forth Cycle", fontsize=16)
    
    plt.plot(t_eval, KE_arr, label='Kinetic Energy (Robot)', color='orange', linewidth=2)
    plt.plot(t_eval, PE_robot_arr - PE_robot_arr[0], label='Gravitational PE (Robot)', color='blue', linewidth=2)
    plt.plot(t_eval, PE_spring_arr, label='Elastic PE (Spring)', color='green', linewidth=2)
    
    Total_E = KE_arr + (PE_robot_arr - PE_robot_arr[0]) + PE_spring_arr
    plt.plot(t_eval, Total_E, label='Total Mechanical Energy', color='black', linestyle='--', linewidth=2)
    
    # Add a vertical line to show where the return trip starts
    plt.axvline(x=T_total, color='red', linestyle=':', label='Turnaround Point')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    
    # ==========================================
    # 8. 3D END-EFFECTOR ANIMATION (FIGURE 3)
    # ==========================================
    from matplotlib.animation import FuncAnimation
    print("Generating 3D Animation...")

    # 1. Pre-calculate the 3D position and rotation of the End Effector (Using the true URDF Frame)
    EE_pos = np.zeros((2 * N_steps, 3))
    EE_rot = np.zeros((2 * N_steps, 3, 3))
    
    for i in range(2 * N_steps):
        q = q_opt[i]
        pin.framesForwardKinematics(model, data, q)
        # Extract translation (Position) and rotation matrix (Orientation) of the tool frame
        EE_pos[i] = data.oMf[tool_frame_id].translation
        EE_rot[i] = data.oMf[tool_frame_id].rotation

    # 2. Setup the 3D Figure
    fig3 = plt.figure(figsize=(10, 8))
    fig3.canvas.manager.set_window_title('End Effector Animation')
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Draw the static path trace
    ax3.plot(EE_pos[:, 0], EE_pos[:, 1], EE_pos[:, 2], color='black', linestyle=':', alpha=0.5)

    # Create empty line objects for the 3 moving axes (The Triad)
    axis_len = 0.15 # Length of the coordinate frame lines in meters
    x_line, = ax3.plot([], [], [], color='red', linewidth=3, label='X (Tool Forward)')
    y_line, = ax3.plot([], [], [], color='green', linewidth=3, label='Y (Tool Left)')
    z_line, = ax3.plot([], [], [], color='blue', linewidth=3, label='Z (Tool Up)')

    # 3. Animation Functions
    def init():
        # Set 3D axis limits based on the trajectory bounds
        pad = 0.3
        ax3.set_xlim([np.min(EE_pos[:, 0]) - pad, np.max(EE_pos[:, 0]) + pad])
        ax3.set_ylim([np.min(EE_pos[:, 1]) - pad, np.max(EE_pos[:, 1]) + pad])
        ax3.set_zlim([np.min(EE_pos[:, 2]) - pad, np.max(EE_pos[:, 2]) + pad])
        ax3.set_xlabel('Global X (m)')
        ax3.set_ylabel('Global Y (m)')
        ax3.set_zlabel('Global Z (m)')
        ax3.set_title('End Effector Pose (Pick & Place Cycle)')
        ax3.legend()
        return x_line, y_line, z_line

    def update(frame):
        p = EE_pos[frame]
        R = EE_rot[frame]

        # Calculate the end points of the triad lines based on current rotation
        px = p + R @ np.array([axis_len, 0, 0])
        py = p + R @ np.array([0, axis_len, 0])
        pz = p + R @ np.array([0, 0, axis_len])

        # Update X-axis (Red)
        x_line.set_data([p[0], px[0]], [p[1], px[1]])
        x_line.set_3d_properties([p[2], px[2]])

        # Update Y-axis (Green)
        y_line.set_data([p[0], py[0]], [p[1], py[1]])
        y_line.set_3d_properties([p[2], py[2]])

        # Update Z-axis (Blue)
        z_line.set_data([p[0], pz[0]], [p[1], pz[1]])
        z_line.set_3d_properties([p[2], pz[2]])

        return x_line, y_line, z_line

    # 4. Execute the animation (Must be assigned to a variable like 'ani'!)
    # interval is in milliseconds. dt * 1000 makes it play in real-time.
    ani = FuncAnimation(fig3, update, frames=2 * N_steps, init_func=init, blit=False, interval=dt * 1000)
    # Finally, show all figures
    plt.show()

    # ==========================================
    # 9. EXPORT OPTIMIZED PARAMETERS
    # ==========================================
    import json
    import os

    print("\nExporting optimized parameters for Isaac Sim...")
    
    # Pack the results into a lightweight JSON
    export_data = {
        "hardware": {
            "stiffness_k": float(result.x[0]),
            "rest_length_L0": float(result.x[1]),
            "ground_anchor_global": result.x[2:5].tolist(),
            "robot_anchor_local": result.x[5:8].tolist()
        },
        "trajectory_control_points": {
            "interior_points": result.x[8:].reshape(6, 4).tolist()
        },
        "task_params": {
            "q_start": q_start.tolist(),
            "q_mid": q_mid.tolist(),
            "q_end": q_end.tolist(),
            "T_total": T_total
        }
    }
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ur5_optimized_params.json")
    with open(save_path, "w") as f:
        json.dump(export_data, f, indent=4)
        
    print(f"Successfully saved optimized params to: {save_path}")


    with open("sim_mujoco03.py", "r") as file:
        exec(file.read())