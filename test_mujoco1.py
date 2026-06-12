import mujoco
import mujoco.viewer
import os
import time

# ==========================================
# 1. LOAD THE MODEL
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "ur5.urdf")
model = mujoco.MjModel.from_xml_path(urdf_path)
data = mujoco.MjData(model)

# Strip URDF physics quirks (disable collisions and hardcoded friction)
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT 
for i in range(model.njnt):
    model.dof_damping[i] = 0.0
    model.dof_frictionloss[i] = 0.0
    model.dof_armature[i] = 0.0

# ==========================================
# 2. SET STATIC TARGETS
# ==========================================
joint_names = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]

# A standard "bent arm" configuration
target_angles = [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]

# Initialize the robot exactly at the target so it doesn't violently snap on frame 1
for name, target in zip(joint_names, target_angles):
    joint = model.joint(name)
    data.qpos[joint.qposadr] = target

# Simple PD Gains
kp = 500.0
kv = 50.0

print("Starting minimal static simulation. Close the window to exit.")

# ==========================================
# 3. MINIMAL SIMULATION LOOP
# ==========================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Clear applied forces
        data.qfrc_applied[:] = 0.0

        # Apply PD Control to hold the position
        for name, target in zip(joint_names, target_angles):
            joint = model.joint(name)
            q_idx = joint.qposadr
            v_idx = joint.dofadr
            
            # Error calculation
            error = target - data.qpos[q_idx]
            vel_error = 0.0 - data.qvel[v_idx]
            
            # Torque = PD + Gravity Compensation
            torque = (kp * error) + (kv * vel_error)
            data.qfrc_applied[v_idx] = torque + data.qfrc_bias[v_idx]

        # Step the physics engine once
        mujoco.mj_step(model, data)
        viewer.sync()

        # Keep the simulation running in real-time
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)