#!/usr/bin/env python3
"""Replay an Odin1 recording in Rerun: colored PCD map + camera trajectory + camera image.

Inputs (recorded with `odin1_ros2.launch.py pcd:=true record_bag:=true`):
  rgb_map_<stamp>.pcd        colored map from pcd_map_saver_node (odom frame)
  camera_traj_<stamp>/       rosbag with /tf (odom -> camera_0) and /odin1/image/compressed
  calib.yaml                 optional, camera intrinsics; enables undistortion and a correct frustum

Usage:
  python3 camera_trajectory.py                      # newest bag + matching PCD, opens the viewer
  python3 camera_trajectory.py --bag DIR --pcd FILE
  python3 camera_trajectory.py --save out.rrd       # write a file instead (open later with `rerun out.rrd`)
  python3 camera_trajectory.py --connect rerun+http://<pc-ip>:9876/proxy   # stream to a remote viewer

Requires: pip install rerun-sdk rosbags numpy opencv-python
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

SCRIPT_DIR = Path(__file__).resolve().parent
ROBOT_MAP_DIR = SCRIPT_DIR.parent / "map"  # {ws}/src/odin_ros_driver/map on the Jetson
IMAGE_TOPIC = "/odin1/image/compressed"
CAMERA_FRAME = "camera_0"


def find_inputs(args):
    """Default to the newest camera_traj_* bag and the PCD with the same timestamp."""
    search = [Path.cwd(), SCRIPT_DIR, ROBOT_MAP_DIR / "bag", ROBOT_MAP_DIR / "pcd"]
    bag = Path(args.bag) if args.bag else None
    if bag is None:
        bags = [p for d in search if d.is_dir() for p in d.glob("camera_traj_*") if p.is_dir()]
        if not bags:
            sys.exit("No camera_traj_* bag found; pass --bag")
        bag = max(bags, key=lambda p: p.name)
    pcd = Path(args.pcd) if args.pcd else None
    if pcd is None:
        stamp = bag.name.removeprefix("camera_traj_")
        pcds = [p for d in search + [bag.parent] if d.is_dir() for p in d.glob("rgb_map_*.pcd")]
        same = [p for p in pcds if p.stem == f"rgb_map_{stamp}"]
        if same:
            pcd = same[0]
        elif pcds:
            pcd = max(pcds, key=lambda p: p.name)
            print(f"warning: no rgb_map_{stamp}.pcd, using newest PCD {pcd.name}")
    calib = Path(args.calib) if args.calib else None
    if calib is None:
        candidates = [bag / "calib.yaml", bag.parent / "calib.yaml", Path.cwd() / "calib.yaml",
                      SCRIPT_DIR / "calib.yaml", Path.home() / ".ros/odin_ros_driver/calib.yaml"]
        calib = next((c for c in candidates if c.is_file()), None)
    return bag, pcd, calib


def load_pcd(path, max_points):
    """Read a binary/ascii PCD with fields x y z rgb (rgb packed as 0x00RRGGBB)."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii").strip()
            if line and not line.startswith("#"):
                key, *vals = line.split()
                header[key] = vals
            if line.startswith("DATA"):
                break
        fields, sizes, types = header["FIELDS"], header["SIZE"], header["TYPE"]
        kinds = {("F", "4"): "<f4", ("F", "8"): "<f8", ("U", "4"): "<u4", ("U", "1"): "u1", ("I", "4"): "<i4"}
        dtype = np.dtype([(n, kinds[(t, s)]) for n, s, t in zip(fields, sizes, types)])
        n = int(header["POINTS"][0])
        if header["DATA"][0] == "binary":
            data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
        elif header["DATA"][0] == "ascii":
            data = np.loadtxt(f, dtype=dtype)
        else:
            sys.exit(f"Unsupported PCD encoding: {header['DATA'][0]}")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = None
    if "rgb" in fields:
        packed = data["rgb"].view(np.uint32) if data["rgb"].dtype != np.uint32 else data["rgb"]
        rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1).astype(np.uint8)
    ok = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[ok], (rgb[ok] if rgb is not None else None)
    if max_points and len(xyz) > max_points:
        keep = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[keep], (rgb[keep] if rgb is not None else None)
    return xyz, rgb


def load_calib(path):
    """Parse cam_0 intrinsics (FishPoly model) from the Odin calib.yaml."""
    text = path.read_text()
    cam = text[text.index("cam_0:"):]
    num = lambda key: float(re.search(rf"^\s*{key}:\s*([-+\d.eE]+)", cam, re.M).group(1))
    return {
        "w": int(num("image_width")), "h": int(num("image_height")),
        "fx": num("A11"), "skew": num("A12"), "fy": num("A22"), "cx": num("u0"), "cy": num("v0"),
        "k": [num(f"k{i}") for i in range(2, 8)],
    }


def undistort_maps(c):
    """Same remap as the driver's buildUndistortMap(): output pinhole pixel -> raw fisheye pixel."""
    u, v = np.meshgrid(np.arange(c["w"], dtype=np.float64), np.arange(c["h"], dtype=np.float64))
    y = (v - c["cy"]) / c["fy"]
    x = (u - c["cx"]) / c["fx"] - y * c["skew"] / c["fx"]
    r = np.maximum(np.hypot(x, y), 1e-8)
    theta = np.arctan(r)
    thetad = theta + sum(k * theta ** (i + 2) for i, k in enumerate(c["k"]))
    xd, yd = x * thetad / r, y * thetad / r
    map_x = (xd * c["fx"] + yd * c["skew"] + c["cx"]).astype(np.float32)
    map_y = (yd * c["fy"] + c["cy"]).astype(np.float32)
    return map_x, map_y


def read_bag(bag_path, image_every):
    """Return camera poses [(t, xyz, quat_xyzw)] from /tf and images [(t, jpeg_bytes)]."""
    poses, images, n_img = [], [], 0
    with AnyReader([bag_path], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        conns = [c for c in reader.connections if c.topic in ("/tf", IMAGE_TOPIC)]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/tf":
                for tf in msg.transforms:
                    if tf.child_frame_id == CAMERA_FRAME:
                        t, q = tf.transform.translation, tf.transform.rotation
                        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                        poses.append((stamp, [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]))
            else:
                if n_img % image_every == 0:
                    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    images.append((stamp, bytes(msg.data)))
                n_img += 1
    poses.sort(key=lambda p: p[0])
    images.sort(key=lambda p: p[0])
    return poses, images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", help="camera_traj_* bag directory (default: newest found)")
    ap.add_argument("--pcd", help="rgb_map_*.pcd (default: same timestamp as the bag)")
    ap.add_argument("--calib", help="calib.yaml (default: next to bag, cwd, script dir, ~/.ros/odin_ros_driver)")
    ap.add_argument("--raw", action="store_true", help="show the raw fisheye JPEG instead of undistorting it")
    ap.add_argument("--image-every", type=int, default=1, help="use every Nth image (default 1)")
    ap.add_argument("--max-points", type=int, default=5_000_000, help="randomly subsample the map above this")
    ap.add_argument("--point-size", type=float, default=0.01, help="map point radius in meters")
    ap.add_argument("--save", help="write an .rrd file instead of opening the viewer")
    ap.add_argument("--connect", help="stream to a running viewer, e.g. rerun+http://IP:9876/proxy")
    args = ap.parse_args()

    bag, pcd, calib_path = find_inputs(args)
    print(f"bag:   {bag}\npcd:   {pcd}\ncalib: {calib_path or 'not found (raw images, approximate frustum)'}")

    poses, images = read_bag(bag, max(1, args.image_every))
    if not poses:
        sys.exit(f"No odom -> {CAMERA_FRAME} transforms in {bag}/tf")
    print(f"{len(poses)} camera poses, {len(images)} images, {poses[-1][0] - poses[0][0]:.1f} s")

    calib = load_calib(calib_path) if calib_path else None
    maps = None
    if calib and not args.raw:
        import cv2
        maps = undistort_maps(calib)

    layout = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="Map + trajectory"),
        rrb.Spatial2DView(origin="/world/camera/image", name="Camera"),
        column_shares=[3, 2],
    )
    rr.init("odin_camera_trajectory", default_blueprint=layout)
    if args.save:
        rr.save(args.save)
    elif args.connect:
        rr.connect_grpc(args.connect)
    else:
        rr.spawn()

    # Static scene: odom frame is ROS convention (x forward, z up)
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if pcd:
        xyz, rgb = load_pcd(pcd, args.max_points)
        print(f"{len(xyz)} map points")
        rr.log("/world/map", rr.Points3D(xyz, colors=rgb, radii=args.point_size), static=True)
    path = np.array([p[1] for p in poses], dtype=np.float32)
    rr.log("/world/trajectory", rr.LineStrips3D([path], colors=[255, 140, 0], radii=0.02), static=True)

    # Camera model: optical frame (x right, y down, z forward); undistorted images are pinhole
    if calib:
        rr.log("/world/camera/image", rr.Pinhole(
            focal_length=[calib["fx"], calib["fy"]], principal_point=[calib["cx"], calib["cy"]],
            width=calib["w"], height=calib["h"], camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.3), static=True)
    else:
        rr.log("/world/camera/image", rr.Pinhole(
            fov_y=1.4, aspect_ratio=1600 / 1296, width=1600, height=1296,
            camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.3), static=True)

    # Timeline is seconds since the first pose (device clock, shared by /tf and images)
    t0 = poses[0][0]
    traveled_step = max(1, len(poses) // 500)  # bound the re-logged path for long recordings
    for i, (t, xyz_c, q) in enumerate(poses):
        rr.set_time("time", duration=t - t0)
        rr.log("/world/camera", rr.Transform3D(translation=xyz_c, quaternion=rr.Quaternion(xyzw=q)))
        if i % traveled_step == 0 or i == len(poses) - 1:
            rr.log("/world/traveled", rr.LineStrips3D([path[: i + 1]], colors=[0, 200, 255], radii=0.03))
    for t, jpeg in images:
        if t < t0:
            continue  # no pose yet for frames before the first /tf
        rr.set_time("time", duration=t - t0)
        if maps is None:
            rr.log("/world/camera/image", rr.EncodedImage(contents=jpeg, media_type="image/jpeg"))
        else:
            raw = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            undist = cv2.remap(raw, maps[0], maps[1], cv2.INTER_LINEAR)
            ok, enc = cv2.imencode(".jpg", undist, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rr.log("/world/camera/image", rr.EncodedImage(contents=enc.tobytes(), media_type="image/jpeg"))
    print("done" + (f", saved {args.save}" if args.save else ""))


if __name__ == "__main__":
    main()
