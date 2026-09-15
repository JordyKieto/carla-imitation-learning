"""Raw recordings -> processed TFRecords (96x96 vision) + Parquet window manifest.

    python preprocess.py                       # recordings/ -> datasets/
    python preprocess.py --pull --push         # same, syncing raw/processed/manifest with S3

Processed files are cached: re-running only processes new recordings, then
rebuilds the manifest from every processed file.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.records import VISION_SHAPE, parse_raw, serialize_processed
from dataset.windows import direction_from_steer, frame_quality, window_rows
from utils import storage
from utils.config import CONFIG, resolve


def recording_tags(raw_path):
    """Tags used to filter training data: map name (e.g. Town01) and source script."""
    stem = os.path.splitext(raw_path)[0]
    tags = []
    if os.path.exists(stem + ".json"):
        with open(stem + ".json") as f:
            map_name = json.load(f).get("map")
        if map_name:
            tags.append(os.path.basename(str(map_name)))
    tags.append("collect" if os.path.exists(stem + ".route.json") else "drive")
    return tags


def process_recording(raw_path, processed_dir):
    stem = Path(raw_path).stem
    processed_path = processed_dir / f"{stem}.processed.tfrecord"
    meta_path = processed_dir / f"{stem}.processed.meta.json"
    if processed_path.exists() and meta_path.exists():
        print(f"[PROCESS] cached {processed_path.name}")
        return processed_path, meta_path

    frames = []
    with tf.io.TFRecordWriter(str(processed_path)) as writer:
        for proto in tf.data.TFRecordDataset(str(raw_path)):
            vision, lidar, telemetry, waypoints, controls = (t.numpy() for t in parse_raw(proto))
            writer.write(serialize_processed(vision, lidar, telemetry, waypoints, controls))
            frames.append({
                "speed": float(telemetry[0]),
                "throttle": float(controls[0]),
                "brake": float(controls[1]),
                "steer": float(controls[2]),
                "direction": direction_from_steer(float(controls[2])),
                "quality": frame_quality(vision, lidar, telemetry, waypoints, controls, VISION_SHAPE),
            })

    meta = {
        "source_file": os.path.basename(raw_path),
        "processed_file": processed_path.name,
        "tags": recording_tags(raw_path),
        "num_frames": len(frames),
        "frames": frames,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[PROCESS] {Path(raw_path).name} -> {processed_path.name} frames={len(frames)} tags={meta['tags']}")
    return processed_path, meta_path


def build_manifest(processed_dir, manifest_path):
    rows = []
    for meta_path in sorted(processed_dir.glob("*.processed.meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        rows.extend(window_rows(meta["frames"], meta["processed_file"], meta["source_file"], meta["tags"]))
    if not rows:
        raise RuntimeError(f"No windows built from {processed_dir}. Record longer drives?")

    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(manifest_path, index=False)
    print(f"[MANIFEST] {manifest_path} windows={len(df)} files={df['processed_file'].nunique()}")
    print(f"[MANIFEST] directions={df['direction'].value_counts().to_dict()}")
    print(f"[MANIFEST] tags={df['tags'].map(tuple).value_counts().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=CONFIG["data"]["raw_dir"])
    parser.add_argument("--processed-dir", default=CONFIG["data"]["processed_dir"])
    parser.add_argument("--manifest", default=CONFIG["data"]["manifest_path"])
    parser.add_argument("--pull", action="store_true", help="download raw recordings from S3 first")
    parser.add_argument("--push", action="store_true", help="upload processed files + manifest to S3")
    args = parser.parse_args()

    raw_dir, processed_dir, manifest_path = resolve(args.raw_dir), resolve(args.processed_dir), resolve(args.manifest)
    processed_dir.mkdir(parents=True, exist_ok=True)
    s3 = CONFIG["s3"]

    if args.pull:
        for key in storage.list_keys(s3["raw_prefix"], suffixes=(".tfrecord", ".json")):
            storage.download(key, raw_dir / os.path.basename(key))

    raw_paths = sorted(glob.glob(str(raw_dir / "*.tfrecord")))
    if not raw_paths and not any(processed_dir.glob("*.processed.meta.json")):
        raise FileNotFoundError(f"No recordings in {raw_dir}. Record some with drive.py or collect.py.")

    for idx, raw_path in enumerate(raw_paths, start=1):
        print(f"[PROCESS] {idx}/{len(raw_paths)}")
        processed_path, meta_path = process_recording(raw_path, processed_dir)
        if args.push:
            storage.upload(processed_path, storage.key(s3["processed_prefix"], processed_path.name))
            storage.upload(meta_path, storage.key(s3["processed_prefix"], meta_path.name))

    build_manifest(processed_dir, manifest_path)
    if args.push:
        storage.upload(manifest_path, storage.key(s3["manifest_prefix"], manifest_path.name))


if __name__ == "__main__":
    main()
