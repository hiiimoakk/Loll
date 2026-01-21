import subprocess
import sys

# List of all required Python packages for your 3D AI
packages = [
    "fastapi==0.110.0",
    "uvicorn[standard]==0.29.0",
    "numpy==1.26.4",
    "scipy==1.12.0",
    "Pillow==10.2.0",
    "opencv-python-headless==4.9.0.80",
    "torch==2.2.1",
    "torchvision==0.17.1",
    "timm==0.9.16",
    "trimesh==4.1.8",
    "pygltflib==1.16.1",
    "open3d==0.18.0",
    "networkx==3.2.1",
    "shapely==2.0.3",
    "python-multipart==0.0.9",
    "requests==2.31.0"
]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg])

print("🔧 Installing required packages...")
for pkg in packages:
    print(f"Installing {pkg}...")
    install(pkg)

print("✅ All packages installed successfully.")