import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import math

# ==========================================
# 1. LOAD THE MODEL
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "ur5.urdf")
model = mujoco.MjModel.from_xml_path(urdf_path)
data = mujoco.MjData(model)

# Strip URDF physics quirks
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
for i in range(model.njnt):
    model.dof_damping[i] = 0.0
    model.dof_frictionloss[i] = 0.0
    model.dof_armature[i] = 0.0

# ==========================================
# 2. SET UP JOINTS & TARGETS
# ==========================================
joint_names = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]

# Base static angles (bent arm pose)
base_angles = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

for j, name in enumerate(joint_names):
    joint = model.joint(name)
    data.qpos[joint.qposadr] = base_angles[j]

# THE FIX: Mass-Scaled PD Gains to prevent Wrist 3 (DOF 5) from exploding
# Shoulders get 500/50, Elbow gets 300/30, Wrists drop drastically to 10/1
kp = np.array([50.0, 50.0, 30.0, 5.0, 5.0, 1.0])
kv = np.array([ 5.0,  5.0,  3.0,  0.5,  0.5,  0.1])

print("Starting stable 2-joint oscillation. Close window to exit.")

t0 = time.time()

# ==========================================
# 3. MINIMAL MOTION LOOP
# ==========================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        sim_time = step_start - t0

        data.qfrc_applied[:] = 0.0

        targets = list(base_angles)
        
        # Joint 1 (Pan) swings left and right
        targets[0] = 0.5 * math.sin(sim_time * 2.0) 
        
        # Joint 2 (Lift) bobs up and down slightly
        targets[1] = -1.57 + 0.3 * math.cos(sim_time * 2.0) 

        for j, name in enumerate(joint_names):
            joint = model.joint(name)
            q_idx = joint.qposadr
            v_idx = joint.dofadr
            
            error = targets[j] - data.qpos[q_idx]
            vel_error = 0.0 - data.qvel[v_idx] 
            
            # Use the specific gain for this specific joint's mass
            torque = (kp[j] * error) + (kv[j] * vel_error)
            data.qfrc_applied[v_idx] = torque + data.qfrc_bias[v_idx]

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)