import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim

# Assume 'ur5_robot' is your loaded Articulation and 'wrist_link' is the RigidPrim 
# where the spring is attached (e.g., Wrist 2).
# Assume 'data' is the loaded JSON dictionary.

current_step = 0
dt = data["dt"]
k = data["hardware"]["stiffness_k"]
L0 = data["hardware"]["rest_length_L0"]
P_g = np.array(data["hardware"]["ground_anchor_global"])

def on_physics_step(step_size):
    global current_step
    
    if current_step >= data["N_steps_total"]:
        return # Cycle complete

    # 1. COMMAND THE MOTORS (Trajectory Tracking)
    # Grab the target angles for this specific time step
    target_q = np.array(data["trajectory"]["positions"][current_step])
    
    # Send to the Isaac Sim PD controller
    action = ArticulationAction(joint_positions=target_q)
    ur5_robot.get_articulation_controller().apply_action(action)
    
    # 2. INJECT THE SPRING FORCE
    # Get the current global position of the wrist link from PhysX
    current_wrist_pos, _ = wrist_link.get_world_pose()
    
    # (Optional: Add the local offset if your anchor isn't exactly at the link origin)
    # current_anchor_pos = current_wrist_pos + local_offset
    
    # Calculate distance and 3D force vector (just like in Pinocchio)
    dist_vec = P_g - current_wrist_pos
    L = np.linalg.norm(dist_vec)
    
    if L > L0:
        # Calculate force magnitude and vector direction
        force_mag = k * (L - L0)
        force_3d = force_mag * (dist_vec / L)
        
        # INJECT force directly into the PhysX rigid body
        wrist_link.apply_forces_and_torques(
            forces=force_3d, 
            positions=current_wrist_pos, 
            is_global=True
        )
        
    current_step += 1