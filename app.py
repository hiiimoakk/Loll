import io
import cv2
import numpy as np
import trimesh
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from scipy.ndimage import gaussian_filter
import tempfile
import os

app = FastAPI(title="UltraMath-3D")

# -----------------------------
# Core math-based depth engine
# -----------------------------

def compute_depth(image_gray):
    h, w = image_gray.shape

    # Normalize
    img = image_gray.astype(np.float32) / 255.0

    # Gradients (structure)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Curvature (shape hint)
    lap = cv2.Laplacian(img, cv2.CV_32F)

    # Local contrast
    blur = gaussian_filter(img, sigma=3)
    contrast = np.abs(img - blur)

    # Depth composition (weighted, stable)
    depth = (
        0.55 * grad_mag +
        0.30 * contrast +
        0.15 * np.abs(lap)
    )

    # Smooth but preserve structure
    depth = gaussian_filter(depth, sigma=1.2)

    # Normalize depth
    depth -= depth.min()
    depth /= (depth.max() + 1e-6)

    # Nonlinear boost for detail
    depth = np.power(depth, 0.7)

    return depth


# --------------------------------
# Point cloud with camera geometry
# --------------------------------

def depth_to_points(depth):
    h, w = depth.shape
    points = []

    # Camera model (approx)
    f = 0.9 * max(h, w)
    cx, cy = w / 2, h / 2

    for y in range(0, h, 2):
        for x in range(0, w, 2):
            z = depth[y, x]
            if z < 0.02:
                continue

            X = (x - cx) * z / f
            Y = (y - cy) * z / f
            Z = z

            points.append([X, -Y, Z])

    return np.array(points)


# -------------------------
# Mesh reconstruction
# -------------------------

def points_to_mesh(points):
    cloud = trimesh.points.PointCloud(points)

    mesh = cloud.convex_hull
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)

    return mesh


# -------------------------
# FastAPI endpoint
# -------------------------

@app.post("/image-to-3d")
async def image_to_3d(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Invalid image"}

    depth = compute_depth(img)
    points = depth_to_points(depth)

    if len(points) < 500:
        return {"error": "Not enough geometry detected"}

    mesh = points_to_mesh(points)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    mesh.export(tmp.name)
    tmp.close()

    return FileResponse(
        tmp.name,
        media_type="model/gltf-binary",
        filename="model.glb"
    )