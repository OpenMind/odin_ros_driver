#!/usr/bin/env python3
"""Rebuild an RGB PCD map from the /odin1/cloud_slam messages in a recorded bag.

Same result as pcd_map_saver_node (voxel grid, mean position and color per voxel),
but done offline and out-of-core: clouds are pre-merged in batches, spilled to
per-tile files on disk, then each tile is merged on its own. RAM stays at a few
hundred MB no matter how long the recording is.

Inputs (recorded with `odin1_ros2.launch.py record_bag:=true`):
  map/bag/camera_traj_<stamp>/    rosbag2 directory containing /odin1/cloud_slam

Usage (source ROS 2 first, for rosbag2_py):
  python3 rosbag2pcd.py file_name:=20261014_124342 voxel_size:=0.01
  python3 rosbag2pcd.py file_name:=/path/to/bag_dir voxel_size:=0.02 output:=/tmp/map.pcd
  (--file_name 20261014_124342 --voxel_size 0.01 also works)

Output: map/pcd/rgb_map_<stamp>_bag_<voxel>m.pcd (binary PCD, FIELDS x y z rgb)
"""
import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_DIR = SCRIPT_DIR.parent / "map"  # {ws}/src/odin_ros_driver/map
BAG_PREFIX = "camera_traj_"

# Same 21-bit-per-axis key packing as pcd_map_saver_node
KEY_OFFSET = 1 << 20
KEY_MASK = (1 << 21) - 1

FLUSH_POINTS = 5_000_000   # raw points buffered before a pre-merge + spill to tiles
MERGE_CHUNK = 4_000_000    # spilled records read per step when merging one tile

# One pre-merged voxel: mean position/color plus how many raw points it stands for
VOXEL_DTYPE = np.dtype([("key", "<i8"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", "<f4"), ("g", "<f4"), ("b", "<f4"), ("n", "<u4")])
PCD_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])


def parse_args():
    # Accept ROS-style key:=value as well as --key value
    argv = []
    for a in sys.argv[1:]:
        if ":=" in a:
            k, v = a.split(":=", 1)
            argv += ["--" + k.lstrip("-"), v]
        else:
            argv.append(a)
    p = argparse.ArgumentParser(description="Rebuild an RGB PCD map from /odin1/cloud_slam in a rosbag.")
    p.add_argument("--file_name", required=True,
                   help="bag stamp (20261014_124342), bag folder name, or path to a bag directory")
    p.add_argument("--voxel_size", type=float, default=0.05, help="voxel size in meters (default 0.05)")
    p.add_argument("--bag_dir", default=str(MAP_DIR / "bag"), help="where bags are looked up by name")
    p.add_argument("--output", default="", help="output .pcd path (default map/pcd/rgb_map_<stamp>_bag_<voxel>m.pcd)")
    p.add_argument("--topic", default="/odin1/cloud_slam")
    p.add_argument("--tile_size", type=float, default=10.0, help="tile edge in meters for out-of-core merging")
    args = p.parse_args(argv)
    if args.voxel_size <= 0:
        p.error("voxel_size must be > 0")
    return args


def resolve_bag(name, bag_dir):
    for cand in (Path(name), Path(bag_dir) / name, Path(bag_dir) / (BAG_PREFIX + name)):
        if cand.is_dir():
            return cand.resolve()
    sys.exit(f"Bag not found: tried {name}, {bag_dir}/{name}, {bag_dir}/{BAG_PREFIX}{name}")


def bag_files(bag):
    """Split files in recording order: from metadata.yaml, or by index suffix if it is missing (crashed recording)."""
    if (bag / "metadata.yaml").exists():
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
                    rosbag2_py.ConverterOptions("", ""))
        files = [bag / p for p in reader.get_metadata().relative_file_paths]
    else:
        files = sorted((p for p in bag.iterdir() if p.suffix in (".mcap", ".db3")),
                       key=lambda p: int(re.search(r"_(\d+)$", p.stem).group(1))
                       if re.search(r"_(\d+)$", p.stem) else 0)
    files = [f for f in files if f.exists()]
    if not files:
        sys.exit(f"No .mcap/.db3 files in {bag}")
    return files


def fmt_duration(sec):
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m"
    if sec >= 60:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec}s"


def cloud_dtype(msg):
    fields = {f.name: f for f in msg.fields}
    missing = [n for n in ("x", "y", "z", "rgb") if n not in fields]
    if missing:
        sys.exit(f"{msg.header.frame_id} cloud has no {missing} field(s)")
    return np.dtype({"names": ["x", "y", "z", "rgb"],
                     "formats": ["<f4", "<f4", "<f4", "<u4"],
                     "offsets": [fields[n].offset for n in ("x", "y", "z", "rgb")],
                     "itemsize": msg.point_step})


def merge(rec):
    """Collapses records sharing a voxel key into one count-weighted mean."""
    keys, inv = np.unique(rec["key"], return_inverse=True)
    w = rec["n"].astype(np.float64)
    total = np.bincount(inv, weights=w)
    out = np.empty(len(keys), VOXEL_DTYPE)
    out["key"] = keys
    out["n"] = np.minimum(total, np.iinfo(np.uint32).max)
    for f in ("x", "y", "z", "r", "g", "b"):
        out[f] = np.bincount(inv, weights=rec[f] * w) / total
    return out


class TileSpiller:
    """Pre-merges buffered points and appends them to one file per spatial tile."""

    def __init__(self, tmp_dir, leaf, tile_size):
        self.tmp_dir = tmp_dir
        self.leaf = leaf
        self.tile_cells = max(1, int(round(tile_size / leaf)))
        self.buf = []
        self.buffered = 0

    def add(self, xyz, rgb):
        self.buf.append((xyz, rgb))
        self.buffered += len(xyz)
        if self.buffered >= FLUSH_POINTS:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        xyz = np.concatenate([b[0] for b in self.buf])
        rgb = np.concatenate([b[1] for b in self.buf])
        self.buf, self.buffered = [], 0

        cells = np.floor(xyz / self.leaf).astype(np.int64)
        keys = (((cells[:, 0] + KEY_OFFSET) & KEY_MASK) << 42) | \
               (((cells[:, 1] + KEY_OFFSET) & KEY_MASK) << 21) | \
               ((cells[:, 2] + KEY_OFFSET) & KEY_MASK)
        rec = np.empty(len(keys), VOXEL_DTYPE)
        rec["key"] = keys
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"] = (rgb >> 16) & 0xff
        rec["g"] = (rgb >> 8) & 0xff
        rec["b"] = rgb & 0xff
        rec["n"] = 1
        rec = merge(rec)

        # Recover cells from the merged keys to bucket each voxel into its tile
        cx = ((rec["key"] >> 42) & KEY_MASK) - KEY_OFFSET
        cy = ((rec["key"] >> 21) & KEY_MASK) - KEY_OFFSET
        cz = (rec["key"] & KEY_MASK) - KEY_OFFSET
        t = self.tile_cells
        tiles = np.stack([cx // t, cy // t, cz // t], axis=1)
        tile_ids, inv = np.unique(tiles, axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(tile_ids) + 1))
        for i, (tx, ty, tz) in enumerate(tile_ids):
            part = rec[order[bounds[i]:bounds[i + 1]]]
            with open(self.tmp_dir / f"tile_{tx}_{ty}_{tz}.bin", "ab") as f:
                part.tofile(f)


def merge_tile(path):
    """Merges one tile file in bounded chunks; memory ~ distinct voxels in the tile."""
    acc = np.empty(0, VOXEL_DTYPE)
    total = path.stat().st_size // VOXEL_DTYPE.itemsize
    for start in range(0, total, MERGE_CHUNK):
        chunk = np.fromfile(path, VOXEL_DTYPE, count=MERGE_CHUNK, offset=start * VOXEL_DTYPE.itemsize)
        acc = merge(np.concatenate([acc, chunk]))
    return acc


def main():
    args = parse_args()
    bag = resolve_bag(args.file_name, args.bag_dir)
    stamp = bag.name[len(BAG_PREFIX):] if bag.name.startswith(BAG_PREFIX) else bag.name
    out_path = Path(args.output) if args.output else \
        MAP_DIR / "pcd" / f"rgb_map_{stamp}_bag_{args.voxel_size:g}m.pcd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f".rosbag2pcd_{stamp}_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()

    files = bag_files(bag)
    total_bytes = sum(f.stat().st_size for f in files) or 1
    print(f"Bag:    {bag} ({len(files)} files, {total_bytes / 1e9:.1f} GB)\n"
          f"Topic:  {args.topic}\nVoxel:  {args.voxel_size} m\nOutput: {out_path}")

    t0 = time.time()
    try:
        # Pass 1: read clouds file by file, pre-merge in batches, spill to tile files
        spiller = TileSpiller(tmp_dir, args.voxel_size, args.tile_size)
        dtype, clouds, raw, done_bytes = None, 0, 0, 0
        width = len(str(len(files)))
        for i, path in enumerate(files, 1):
            print(f"[{i:>{width}}/{len(files)}] processing {path.name} ...", flush=True)
            reader = rosbag2_py.SequentialReader()
            reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=""),
                        rosbag2_py.ConverterOptions("", ""))
            reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
            file_clouds = 0
            while reader.has_next():
                _, data, _ = reader.read_next()
                msg = deserialize_message(data, PointCloud2)
                file_clouds += 1
                n = msg.width * msg.height
                if n == 0:
                    continue
                if dtype is None or dtype.itemsize != msg.point_step:
                    dtype = cloud_dtype(msg)
                if msg.row_step != msg.width * msg.point_step:
                    sys.exit("Clouds with row padding are not supported")
                pts = np.frombuffer(msg.data, dtype, count=n)
                xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
                ok = np.isfinite(xyz).all(axis=1)
                spiller.add(xyz[ok], pts["rgb"][ok])
                raw += int(ok.sum())
            del reader
            clouds += file_clouds
            done_bytes += path.stat().st_size
            elapsed = time.time() - t0
            eta = elapsed * (total_bytes - done_bytes) / done_bytes if done_bytes else 0
            print(f"[{i:>{width}}/{len(files)}] {file_clouds} clouds | total {clouds} clouds, "
                  f"{raw / 1e6:.1f}M points | {fmt_duration(elapsed)} elapsed, ~{fmt_duration(eta)} left",
                  flush=True)
        spiller.flush()
        if clouds == 0:
            sys.exit(f"No {args.topic} messages in {bag} (recorded before cloud_slam was added to the bag?)")
        print(f"[{len(files)}/{len(files)}] ---- Completed reading: {clouds} clouds, {raw / 1e6:.1f}M points "
              f"in {fmt_duration(time.time() - t0)}", flush=True)

        # Pass 2: merge each tile and stream its voxels into the PCD body
        tiles = sorted(tmp_dir.glob("tile_*.bin"))
        body_path = tmp_dir / "body.bin"
        count = 0
        print(f"Merging {len(tiles)} map tiles ...", flush=True)
        with open(body_path, "wb") as body:
            for i, tile in enumerate(tiles, 1):
                vox = merge_tile(tile)
                tile.unlink()
                pts = np.empty(len(vox), PCD_DTYPE)
                pts["x"], pts["y"], pts["z"] = vox["x"], vox["y"], vox["z"]
                rgb = [np.clip(np.rint(vox[c]), 0, 255).astype(np.uint32) for c in ("r", "g", "b")]
                pts["rgb"] = 0xff000000 | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
                pts.tofile(body)
                count += len(pts)
                if i % 50 == 0 or i == len(tiles):
                    print(f"  merged {i}/{len(tiles)} tiles, {count / 1e6:.2f}M voxels "
                          f"({fmt_duration(time.time() - t0)} elapsed)", flush=True)

        if count == 0:
            sys.exit("No valid points found, nothing written")

        # Header needs the final count, so prepend it and copy the body; rename makes it atomic
        part_path = out_path.with_name(out_path.name + ".tmp")
        with open(part_path, "wb") as out, open(body_path, "rb") as body:
            out.write((f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\n"
                       f"SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH {count}\nHEIGHT 1\n"
                       f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {count}\nDATA binary\n").encode())
            shutil.copyfileobj(body, out, 16 << 20)
        os.replace(part_path, out_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {count} points ({size_mb:.1f} MB) to {out_path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
