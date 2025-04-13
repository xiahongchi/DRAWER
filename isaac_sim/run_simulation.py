# Import necessary modules from Omniverse Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import omni.usd
from pxr import UsdGeom, Gf, Usd
import carb
import numpy as np
import os
import json 
import argparse

# Step 1: Load a USD file using provided file path
def load_usd_file(file_path):
    stage = omni.usd.get_context().open_stage(file_path)
    if not stage:
        carb.log_error("Failed to open USD stage.")
        return None
    return stage

# Step 2: Get object according to given USD prim path
def get_prim(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        carb.log_error(f"Prim at path {prim_path} is not valid.")
        return None
    return prim

# Helper function to extract position and orientation from the transformation matrix
def extract_position_orientation(transform):
    position = Gf.Vec3d(transform.ExtractTranslation())
    rotation = transform.ExtractRotationQuat()
    orientation = Gf.Quatd(rotation.GetReal(), *rotation.GetImaginary())
    return position, orientation

# Step 3: Start simulation and trace position, orientation, and speed of the object
def start_simulation_and_trace(prims, duration=5.0, dt=1.0/60.0):
    # Define a list to store the traced data
    traced_data = {}
    
    # Get the timeline and start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    
    # Initialize variables for tracking the previous position for speed calculation
    prev_position = None
    elapsed_time = 0.0
    
    while elapsed_time < duration:
        print(f"\relapsed_time: {elapsed_time:.5f}", end="")
        # Step the simulation
        simulation_app.update()
        
        # Get the current time code
        current_time_code = Usd.TimeCode.Default()
        
        # Get current position and orientation
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)
            
        for prim_i, (position, orientation) in enumerate(traced_data_frame_prims):
        # Calculate speed if previous position is available
            speed = 0.0
            if prev_position is not None:
                distance = np.linalg.norm(position - prev_position)
                speed = distance / dt
            
            # Update previous position
            prev_position = position
            
            # Store the data for current frame
            traced_data_prim = traced_data.get(f"{prim_i:0>2d}", [])
            
            traced_data_prim.append({
                "time": elapsed_time,
                "position": [position[0], position[1], position[2]],
                "orientation": [orientation.GetReal(), 
                                orientation.GetImaginary()[0], 
                                orientation.GetImaginary()[1], 
                                orientation.GetImaginary()[2]
                                ],
                "speed": speed
            })

            traced_data[f"{prim_i:0>2d}"] = traced_data_prim
        
        # Increment the elapsed time
        elapsed_time += dt
        
    # Stop the simulation
    timeline.stop()
    
    return traced_data

def main():
    # Provide the file path to your USD file and the USD prim path
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sdf_dir", type=str, required=True)
    args = parser.parse_args()
    ckpt_dir = args.sdf_dir
    
    usd_file_path = os.path.join(ckpt_dir, "drawers", "joint.usda")
    traced_data_savepath = os.path.join(ckpt_dir, "drawers", "trace.json")
    usd_drawers_dir = os.path.join(ckpt_dir, "drawers", "results")
    
    # Load the USD file
    stage = load_usd_file(usd_file_path)
    if stage is None:
        assert False

    prims = []
    drawer_files = [name for name in os.listdir(usd_drawers_dir) if name.endswith(".ply")]
    total_frame_num = len(drawer_files)
    
    # Get the prim of the object
    for prim_i in range(total_frame_num):
        usd_prim_path = f"/World/drawer/drawer_{prim_i:0>2d}/drawer"
        prim = get_prim(usd_prim_path)
        if prim is None:
            assert False
        prims.append(prim)
    
    # Start the simulation and trace the object
    traced_data = start_simulation_and_trace(prims)
    
    # Print the traced data
    # for data in traced_data:
    #     print(f"Time: {data['time']:.2f}, Position: {data['position']}, Orientation: {data['orientation']}, Speed: {data['speed']:.2f}")
    with open(traced_data_savepath, 'w') as f:
        json.dump(traced_data, f, indent=4)

# Run the main function
if __name__ == "__main__":
    main()

# Shutdown the SimulationApp
simulation_app.close()
