import mujoco
import mujoco.viewer
import numpy as np
import json
import os
import time
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE MODEL & OPTIMIZED PARAMS
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(script_dir, "ur5_optimized_params.json")
with open(json_path, "r") as f:
    opt_data = json.load(f)

k = opt_data["hardware"]["stiffness_k"]
L0 = opt_data["hardware"]["rest_length_L0"]
P_g = np.array(opt_data["hardware"]["ground_anchor_global"])
P_r_local = np.array(opt_data["hardware"]["robot_anchor_local"])
cp_interior = np.array(opt_data["trajectory_control_points"]["interior_points"])

q_start = np.array(opt_data["task_params"]["q_start"])
q_mid = np.array(opt_data["task_params"]["q_mid"])
q_end = np.array(opt_data["task_params"]["q_end"])
T_total = opt_data["task_params"]["T_total"]

# Load URDF
urdf_path = os.path.join(script_dir, "ur5.urdf")
model = mujoco.MjModel.from_xml_path(urdf_path)
data = mujoco.MjData(model)

# Official MuJoCo configuration flags
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

# Clear hardcoded URDF passive limits to allow clean external control
for i in range(model.njnt):
    model.dof_damping[i] = 0.0
    model.dof_frictionloss[i] = 0.0
    model.dof_armature[i] = 0.0

# ==========================================
# 2. HIGH-RESOLUTION TRAJECTORY GENERATION
# ==========================================
# Run spline evaluation at a continuous, steady 1000Hz to prevent step jumps
dt = 0.001  
model.opt.timestep = dt

N_steps = int(T_total / dt)  
t_knots = np.linspace(0, T_total, 7)
t_eval = np.linspace(0, T_total, N_steps)
q_fwd = np.zeros((N_steps, 6))
v_fwd = np.zeros((N_steps, 6))

for j in range(6):
    full_path = np.array([q_start[j], cp_interior[j,0], cp_interior[j,1], q_mid[j], cp_interior[j,2], cp_interior[j,3], q_end[j]])
    spline = CubicSpline(t_knots, full_path, bc_type='clamped')
    q_fwd[:, j] = spline(t_eval)
    v_fwd[:, j] = spline(t_eval, 1)

q_cycle = np.concatenate((q_fwd, q_fwd[::-1]), axis=0)
v_cycle = np.concatenate((v_fwd, -v_fwd[::-1]), axis=0)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_2_link")

# List of joints matching your spline columns
urdf_joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

# --- CRITICAL CRASH FIX: Initialize robot positions to match trajectory start ---
for j, name in enumerate(urdf_joint_names):
    joint_view = model.joint(name)
    data.qpos[joint_view.qposadr] = q_start[j]

# ==========================================
# 3. SIMULATION LOOP WITH NATIVE ACCESSORS
# ==========================================
N_steps_cycle = len(q_cycle)
total_simulation_steps = N_steps_cycle * 2  

target_q_log = np.zeros((total_simulation_steps, 6))
actual_q_log = np.zeros((total_simulation_steps, 6))
time_log = np.zeros(total_simulation_steps)

# Gains tuned to physical UR5 link inertias
# WORKED
# kp = np.array([50.0, 50.0, 30.0, 5.0, 5.0, 1.0])
# kv = np.array([ 5.0,  5.0,  3.0,  0.5,  0.5,  0.1])

kp = np.array([50.0, 150.0, 60.0, 8.0, 7.0, 3.0])
kv = np.array([ 15.0,  60.0,  6.0,  1.0,  0.7,  0.1])


step_index = 0
total_energy_squared_torque = 0.0
sim_time = 0.0

render_fps = 60
render_interval = 1.0 / render_fps
last_render_time = time.time()

print("Starting stable, continuous-time simulation...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running() and step_index < total_simulation_steps:
        step_start = time.time()

        target_q = q_cycle[step_index % N_steps_cycle]
        target_v = v_cycle[step_index % N_steps_cycle]

        target_q_log[step_index] = target_q
        time_log[step_index] = sim_time

        # Reset applied generalized forces before computation
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:, :] = 0.0

        # Official Python accessor pattern for tracking loop
        for j, name in enumerate(urdf_joint_names):
            joint_view = model.joint(name)
            q_idx = joint_view.qposadr
            v_idx = joint_view.dofadr
            
            # Read physical state variables natively
            current_q = data.qpos[q_idx]
            current_v = data.qvel[v_idx]
            
            # Compute PD tracking force + add native gravity feedforward (qfrc_bias)
            torque = kp[j] * (target_q[j] - current_q) + kv[j] * (target_v[j] - current_v)
            data.qfrc_applied[v_idx] = torque + data.qfrc_bias[v_idx]

        # ========================================================
        # UNCOMMENT THIS BLOCK LATER TO RE-INTRODUCE THE SPRING:
        # ========================================================
        # P_r_global = data.xpos[body_id] + data.xmat[body_id].reshape(3,3) @ P_r_local
        # dist_vec = P_g - P_r_global
        # L = np.linalg.norm(dist_vec)
        # if L > L0:
        #     force_mag = k * (L - L0)
        #     force_3d = force_mag * (dist_vec / L)
        #     data.xfrc_applied[body_id, :3] = force_3d
        # ========================================================

        # Step physics engine exactly once per loop step
        mujoco.mj_step(model, data)
        
        # Log actual positions safely
        for j, name in enumerate(urdf_joint_names):
            actual_q_log[step_index, j] = data.qpos[model.joint(name).qposadr]

        total_energy_squared_torque += np.sum(data.qfrc_applied[:6]**2) * dt
        sim_time += dt
        step_index += 1
        
        if (time.time() - last_render_time) > render_interval:
            viewer.sync()
            last_render_time = time.time()

        elapsed = time.time() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

print(f"Total Squared Torque Baseline: {total_energy_squared_torque:.2f}")

# ==========================================
# 4. PLOT TARGET VS ACTUAL TRAJECTORY
# ==========================================
print("Generating tracking plots...")
joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.canvas.manager.set_window_title('Trajectory Tracking Verification')

for j in range(6):
    row = j // 3
    col = j % 3
    axs[row, col].plot(time_log, target_q_log[:, j], 'k--', linewidth=2, label='Target (NLP)')
    axs[row, col].plot(time_log, actual_q_log[:, j], 'r-', linewidth=1.5, label='Actual (MuJoCo)')
    axs[row, col].set_title(joint_names[j])
    axs[row, col].set_xlabel('Time (s)')
    axs[row, col].set_ylabel('Angle (rad)')
    axs[row, col].grid(True, linestyle=':', alpha=0.6)
    if j == 0:
        axs[row, col].legend()

plt.tight_layout()
plt.show()