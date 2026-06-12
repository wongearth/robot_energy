import sys
import os

print("1. Python Executable:", sys.executable)
print("2. Python Version:", sys.version.split()[0])
print("\n3. Looking for packages in these directories:")
for p in sys.path: 
    print(" -", p)

print("\n4. Checking physical folder:")
target_dir = "/home/wong/.env_robot_energy/lib/python3.11/site-packages"
if os.path.exists(target_dir):
    contents = [f for f in os.listdir(target_dir) if 'example' in f.lower()]
    print(f"   Found related files in 3.11 site-packages: {contents}")
else:
    print(f"   Directory does NOT exist: {target_dir}")