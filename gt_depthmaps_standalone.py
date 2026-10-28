"""GT depth maps for a trained DP-GS model, standalone (Python 3.8, needs only numpy + open3d).

Raycasts the TnT GT mesh from every camera of the model and puts the model's rendered depth maps into the same
GT coordinate system, so both can be compared pixel by pixel in GT units (metres for TnT).

    python gt_depthmaps_standalone.py --model_path <model> --gt_dir <folder with <scene>_poisson_mesh.ply + <scene>_trans.txt>

Before training (no model yet), read the cameras from the dataset the same way train_w_uncertainty.py does:
    python gt_depthmaps_standalone.py -s <dataset>/<scene> --gt_dir <gt folder> [--split_file split.json --split test]
which writes <dataset>/<scene>/gt_depth/<view>.npy (or --out_dir).

Coordinate systems: <scene>_trans.txt is the 4x4 similarity [scale * R | t] that maps the model (reconstruction)
frame to the GT frame. Instead of moving the mesh, every camera is moved into the GT frame. A camera-frame z-depth
is then only multiplied by `scale` (rotation and translation move the camera, not the depth along its own axis):
    GT depth       = raycast of the untouched GT mesh from the camera moved into the GT frame
    rendered depth = model depth map * scale
Both are camera-frame z-depth (what DP-GS renders as plane_depth) on the model's pixel grid, NaN = no surface.

Read from the model folder: cameras.json (poses + focal lengths at full image size), cfg_args (--resolution) and
rendered_depth/iteration_<N>/<view>.npy (latest N; or pass --depth_dir; skipped if there are none).
Writes <out_dir>/gt_depth/<view>.npy and <out_dir>/rendered_depth_gt_frame/<view>.npy (default out_dir: <model>).
"""
import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", "-s", default=None,
                    help="dataset folder (the -s of training: sparse/0 + images/). Cameras are read from COLMAP "
                         "exactly like train_w_uncertainty.py, so no trained model is needed")
parser.add_argument("--images", "-i", default="images", help="image folder inside --source_path (training -i)")
parser.add_argument("--model_path", "-m", default=None,
                    help="trained model folder (cameras.json + cfg_args); only needed without --source_path "
                         "or to also convert the model's rendered depth")
parser.add_argument("--gt_dir", required=True, help="folder with <scene>_poisson_mesh.ply and <scene>_trans.txt")
parser.add_argument("--depth_dir", default=None, help="rendered depth maps <view>.npy (default: the latest "
                                                     "<model>/rendered_depth/iteration_N)")
parser.add_argument("--out_dir", default=None, help="default: <model_path>, else <source_path>")
parser.add_argument("--split_file", default=None, help='json like {"train": [names], "test": [names]}; '
                                                      'with --split, only those views are processed')
parser.add_argument("--split", default="test", help="which list of --split_file to use")
parser.add_argument("--cameras_json", default=None,
                    help="camera file in the 3DGS cameras.json format (instead of --source_path)")
parser.add_argument("--resolution", "-r", type=int, default=None,
                    help="training --resolution (1/2/4/8/-1); default: from <model>/cfg_args, else -1")
parser.add_argument("--frame", choices=["colmap", "gt"], default="colmap",
                    help="units of the saved depth: colmap = the COLMAP / training frame (GT depth / scale, same as "
                         "the existing gt_depth_train), gt = TnT GT frame in metres")
args = parser.parse_args()
if not (args.source_path or args.model_path or args.cameras_json):
    raise SystemExit("pass --source_path (dataset) or --model_path (trained model)")
out_dir = args.out_dir or args.model_path or args.source_path

# model frame -> GT frame: x_gt = scale * rotation @ x_model + translation
model_to_gt = np.loadtxt(glob.glob(os.path.join(args.gt_dir, "*_trans.txt"))[0])
scale = np.cbrt(np.linalg.det(model_to_gt[:3, :3])) ## This is becauue scale times orthonormal 3x3 mtx is just scale^3 determinant. 
print("model -> GT scale: {:.4f}, saving depth in the {} frame".format(scale, args.frame))
rotation = model_to_gt[:3, :3] / scale
translation = model_to_gt[:3, 3]

# GT mesh, left in the GT frame
print("loading GT mesh ...")
mesh = o3d.io.read_triangle_mesh(glob.glob(os.path.join(args.gt_dir, "*_poisson_mesh.ply"))[0])
raycaster = o3d.t.geometry.RaycastingScene()
raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# the training image size (utils/camera_utils.py loadCam). cfg_args is the printed argparse.Namespace of the
# training run, read back the way the 3DGS / DP-GS render scripts do
cfg_path = os.path.join(args.model_path, "cfg_args") if args.model_path else ""
if args.resolution is not None:
    resolution = args.resolution
elif os.path.isfile(cfg_path):
    resolution = eval(open(cfg_path).read(), {"Namespace": argparse.Namespace}).resolution
else:
    resolution = -1
print("resolution:", resolution)


def cameras_from_colmap(source_path, images):
    """Cameras as in scene/dataset_readers.py readColmapCameras, in the cameras.json layout."""
    import importlib.util
    # load scene/colmap_loader.py by path: importing the scene package would pull in torch
    spec = importlib.util.spec_from_file_location(
        "colmap_loader", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene", "colmap_loader.py"))
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    sparse = os.path.join(source_path, "sparse", "0")
    # .bin when present, else .txt; a failing .bin read raises its own error instead of falling back
    if os.path.isfile(os.path.join(sparse, "images.bin")):
        extr = cl.read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    else:
        extr = cl.read_extrinsics_text(os.path.join(sparse, "images.txt"))
    if os.path.isfile(os.path.join(sparse, "cameras.bin")):
        intr = cl.read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    else:
        intr = cl.read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    print("COLMAP: {} images, {} cameras from {}".format(len(extr), len(intr), sparse))
    try:
        from PIL import Image
    except ImportError:
        Image = None
    cams = []
    for key in sorted(extr, key=lambda k: extr[k].name):
        e = extr[key]
        c = intr[e.camera_id]
        if c.model == "SIMPLE_PINHOLE":
            fx = fy = c.params[0]
        elif c.model == "PINHOLE":
            fx, fy = c.params[0], c.params[1]
        else:
            raise SystemExit("COLMAP camera model {} not handled (training only supports PINHOLE / SIMPLE_PINHOLE)"
                             .format(c.model))
        # training resizes from the size of the image file, and the FoV comes from the COLMAP width / height
        w, h = c.width, c.height
        img_path = os.path.join(source_path, images, os.path.basename(e.name))
        if Image is not None and os.path.isfile(img_path):
            with Image.open(img_path) as im:
                w, h = im.size
        w2c_rot = cl.qvec2rotmat(e.qvec)
        cams.append({"img_name": os.path.basename(e.name).split(".")[0], "width": w, "height": h,
                     "fx": fx * w / c.width, "fy": fy * h / c.height,
                     "position": (-w2c_rot.T @ np.array(e.tvec)).tolist(), "rotation": w2c_rot.T.tolist()})
    return cams


if args.source_path and not args.cameras_json:
    cameras = cameras_from_colmap(args.source_path, args.images)
    cameras_path = os.path.join(args.source_path, "sparse", "0")
else:
    cameras_path = args.cameras_json or os.path.join(args.model_path, "cameras.json")
    if not os.path.isfile(cameras_path):
        raise SystemExit("no {}; use --source_path to read the cameras from the dataset".format(cameras_path))
    cameras = json.load(open(cameras_path))
    need = ("img_name", "width", "height", "position", "rotation", "fx", "fy")
    if not (isinstance(cameras, list) and cameras and isinstance(cameras[0], dict)
            and all(k in cameras[0] for k in need)):
        raise SystemExit("{} is not a 3DGS camera list (need {} per camera). The official TnT <scene>.json "
                         "is the crop volume, not cameras.".format(cameras_path, need))
print("cameras from:", cameras_path, "({} views)".format(len(cameras)))
if args.split_file:
    wanted = [str(n) for n in json.load(open(args.split_file))[args.split]]
    stem = lambda n: os.path.splitext(os.path.basename(str(n)))[0]  # "000002.jpg" and "000002" both match
    wanted_stems = {stem(n) for n in wanted}
    cameras = [c for c in cameras if stem(c["img_name"]) in wanted_stems]
    missing = sorted(wanted_stems - {stem(c["img_name"]) for c in cameras})
    print("{} split: {} of {} views found".format(args.split, len(cameras), len(wanted)))
    if missing:
        print("  no camera for (skipped):", missing)

depth_dir = args.depth_dir
if depth_dir is None and args.model_path:
    iteration_dirs = glob.glob(os.path.join(args.model_path, "rendered_depth", "iteration_*"))
    if iteration_dirs:
        depth_dir = max(iteration_dirs, key=lambda d: int(d.rsplit("_", 1)[1]))
print("rendered depth from:", depth_dir)
os.makedirs(os.path.join(out_dir, "gt_depth"), exist_ok=True)


for cam in cameras:
    name = cam["img_name"]
    if resolution in (1, 2, 4, 8):
        downscale = resolution
        width, height = round(cam["width"] / downscale), round(cam["height"] / downscale)  # as loadCam
    else:                                   # -1: only images wider than 1600 px are downscaled
        downscale = max(cam["width"] / 1600, 1)
        width, height = int(cam["width"] / downscale), int(cam["height"] / downscale)
    fx = cam["fx"] * width / cam["width"]
    fy = cam["fy"] * height / cam["height"]
    cx, cy = width / 2, height / 2           # DP-GS pins the principal point to the image centre

    # camera -> GT frame (cameras.json holds the camera centre and the camera-to-world rotation)
    center_gt = scale * rotation @ np.array(cam["position"]) + translation
    cam_to_gt_rotation = rotation @ np.array(cam["rotation"])

    # one ray per pixel; the camera-frame direction has z = 1, so the hit distance t is the z-depth itself
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, dtype=np.float64)], -1).reshape(-1, 3)
    dirs_gt = dirs_cam @ cam_to_gt_rotation.T
    rays = np.concatenate([np.broadcast_to(center_gt, dirs_gt.shape), dirs_gt], 1).astype(np.float32)
    t_hit = raycaster.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(height, width)
    gt_depth = np.where(np.isfinite(t_hit), t_hit, np.nan)
    if args.frame == "colmap":
        gt_depth = gt_depth / scale  # a z-depth only scales between the two frames
    gt_depth = gt_depth.astype(np.float32)
    np.save(os.path.join(out_dir, "gt_depth", name + ".npy"), gt_depth)

    msg = "{}: {}x{}, GT mesh hit by {:.1f}% of pixels".format(name, width, height, np.isfinite(gt_depth).mean() * 100)
    depth_file = os.path.join(depth_dir, name + ".npy") if depth_dir else ""
    if os.path.isfile(depth_file):
        rendered = np.load(depth_file).astype(np.float32).squeeze()
        rendered = np.where(rendered > 0, rendered * (scale if args.frame == "gt" else 1.0), np.nan).astype(np.float32)
        # own folder, so it does not mix with the model's rendered_depth/iteration_N inputs
        os.makedirs(os.path.join(out_dir, "rendered_depth_gt_frame"), exist_ok=True)
        np.save(os.path.join(out_dir, "rendered_depth_gt_frame", name + ".npy"), rendered)
        both = np.isfinite(gt_depth) & np.isfinite(rendered)
        d = np.median(np.abs(rendered - gt_depth)[both]) * (1.0 if args.frame == "gt" else scale)
        msg += ", median |rendered - GT| = {:.2f} cm".format(d * 100)
    print(msg)
print("saved to", out_dir)
