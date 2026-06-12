import mujoco
import numpy as np
import json
import os
import time
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import mujoco.viewer

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

# Load the URDF directly
urdf_path = os.path.join(script_dir, "ur5.urdf")
model = mujoco.MjModel.from_xml_path(urdf_path)
data = mujoco.MjData(model)

# Disable mesh collisions so joints don't jam internally
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

for i in range(model.njnt):
    model.dof_damping[i] = 0.0
    model.dof_frictionloss[i] = 0.0
    model.dof_armature[i] = 0.0

# ==========================================
# 2. TRAJECTORY GENERATION FROM SPLINE
# ==========================================
N_steps = 100  # 100 discrete trajectory points over 0.5s (200 Hz update rate)
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

# High-fidelity sub-stepping configuration
dt_trajectory = T_total / N_steps          # 0.005s (Main loop rate)
model.opt.timestep = 0.0005                 # 0.0005s (2000Hz internal physics engine clock)
substeps = int(dt_trajectory / model.opt.timestep) 

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_2_link")

# ==========================================
# 3. SIMULATION LOOP WITH EXPLICIT JOINT MAPPING
# ==========================================
N_steps_cycle = len(q_cycle)
total_simulation_steps = N_steps_cycle * 2  # Log data for exactly 2 cycles

target_q_log = np.zeros((total_simulation_steps, 6))
actual_q_log = np.zeros((total_simulation_steps, 6))
time_log = np.zeros(total_simulation_steps)

# Exact joint names from your URDF
urdf_joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

# Map names to true internal scalar memory indexes
joint_qpos_adrs = []
joint_dof_adrs = []
for name in urdf_joint_names:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jnt_id == -1:
        raise ValueError(f"Joint '{name}' not found in URDF! Check spelling.")
    joint_qpos_adrs.append(model.jnt_qposadr[jnt_id])
    joint_dof_adrs.append(model.jnt_dofadr[jnt_id])

# --- CRITICAL FIX 1: INITIAL STATE SHOCK ---
# Place the robot exactly at the start of the NLP trajectory.
for j in range(6):
    data.qpos[joint_qpos_adrs[j]] = q_start[j]

# Run forward kinematics once before the loop so MuJoCo calculates 
# the accurate gravity bias for this exact starting posture.
mujoco.mj_forward(model, data)

# --- CRITICAL FIX 2: YOUR VALIDATED GAINS ---
# kp = np.array([50.0, 50.0, 30.0, 5.0, 5.0, 1.0])
# kv = np.array([ 5.0,  5.0,  3.0,  0.5,  0.5,  0.1])

kp = np.array([750.0, 1800.0, 460.0, 80.0, 15.0, 20.0])
kv = np.array([ 100.0,  320.0,  80.0,  30.0,  3.0,  0.5])


step_index = 0
total_energy_squared_torque = 0.0
sim_time = 0.0

print("Starting Stable, Unjammed MuJoCo Simulation...")

render_fps = 60
render_interval = 1.0 / render_fps
last_render_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running() and step_index < total_simulation_steps:
        step_start = time.time()

        target_q = q_cycle[step_index % N_steps_cycle]
        target_v = v_cycle[step_index % N_steps_cycle]

        target_q_log[step_index] = target_q
        time_log[step_index] = sim_time

        tau_squared_accum = 0.0
        
        # Execute multiple physics sub-steps per trajectory loop to ensure smoothness
        for _ in range(substeps):
            data.qfrc_applied[:] = 0.0   

            current_q = np.array([data.qpos[adr] for adr in joint_qpos_adrs])
            current_v = np.array([data.qvel[adr] for adr in joint_dof_adrs])
            
            # --- CRITICAL FIX 3: GRAVITY FEEDFORWARD ---
            # Extract gravity & Coriolis forces natively calculated by MuJoCo
            bias_torques = np.array([data.qfrc_bias[adr] for adr in joint_dof_adrs])

            # Compute PD tracking torque
            tau_pd = kp * (target_q - current_q) + kv * (target_v - current_v)
            
            # Combine PD with Gravity Compensation
            tau_total = tau_pd + bias_torques

            # Map calculated torques back to their exact DoF addresses
            for j in range(6):
                data.qfrc_applied[joint_dof_adrs[j]] = tau_total[j]

            # ========================================================
            # OPTIONAL: TO RE-INTRODUCE YOUR SPRING LATER, UNCOMMENT THIS:
            # ========================================================
            # P_r_global = data.xpos[body_id] + data.xmat[body_id].reshape(3,3) @ P_r_local
            # dist_vec = P_g - P_r_global
            # L = np.linalg.norm(dist_vec)
            # if L > L0:
            #     force_mag = k * (L - L0)
            #     force_3d = force_mag * (dist_vec / L)
            #     data.xfrc_applied[body_id, :3] = force_3d
            # else:
            #     data.xfrc_applied[body_id, :3] = 0.0
            # ========================================================

            tau_squared_accum += np.sum(tau_total**2)
            mujoco.mj_step(model, data)
        
        # Log physical positions after sub-steps conclude
        actual_q_log[step_index] = np.array([data.qpos[adr] for adr in joint_qpos_adrs])
        sim_time += dt_trajectory
        
        if (time.time() - last_render_time) > render_interval:
            render_start = time.time()
            viewer.sync()
            last_render_time = render_start

        total_energy_squared_torque += tau_squared_accum * (dt_trajectory / substeps)
            
        step_index += 1
        
        elapsed = time.time() - step_start
        if elapsed < dt_trajectory:
            time.sleep(dt_trajectory - elapsed)

print(f"Simulation completed. Total Squared Torque Baseline: {total_energy_squared_torque:.2f}")

# ==========================================
# 4. PLOT TARGET VS ACTUAL TRAJECTORY
# ==========================================
print("Generating tracking plots...")
joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.canvas.manager.set_window_title('MuJoCo Trajectory Tracking Verification')

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