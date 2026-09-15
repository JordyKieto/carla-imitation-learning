"""Behavioral-cloning training for the multimodal driving policy.

    python train.py                                  # uses config.yaml + datasets/
    python train.py --s3                             # pull manifest/processed from S3, push checkpoints
    python train.py --manifest samples/windows_manifest.parquet --processed-dir samples --epochs 1

Loss: Huber(throttle) + Huber(brake) + MSE(steer) on the last frame of each window.
Logs to Weights & Biases when WANDB_API_KEY is set in .env.
"""
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.records import CONTROLS_SHAPE, LIDAR_SHAPE, TELEMETRY_SHAPE, VISION_SHAPE, WAYPOINTS_SHAPE
from dataset.windows import DIRECTION_NAMES, STACK_SIZE, filter_manifest, make_dataset, split_train_val, to_ranges
from model.metrics import batch_metrics, direction_losses
from model.policy import build_policy, load_policy
from utils import storage
from utils.config import CONFIG, env, resolve


def parse_args():
    cfg = CONFIG["training"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=CONFIG["data"]["manifest_path"])
    parser.add_argument("--processed-dir", default=CONFIG["data"]["processed_dir"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=cfg["learning_rate"])
    parser.add_argument("--val-steps", type=int, default=cfg["val_steps"])
    parser.add_argument("--log-every", type=int, default=cfg["log_every_steps"])
    parser.add_argument("--checkpoint-every", type=int, default=cfg["checkpoint_every_epochs"])
    parser.add_argument("--checkpoint-dir", default=cfg["checkpoint_dir"])
    parser.add_argument("--tags", nargs="*", default=cfg["tags"], help="only train on windows with these tags")
    parser.add_argument("--run-name", default=cfg["run_name"])
    parser.add_argument("--resume", help="path to a .keras checkpoint to continue from")
    parser.add_argument("--s3", action="store_true", help="sync data from and checkpoints to S3")
    return parser.parse_args()


def load_data(args):
    manifest_path, processed_dir = resolve(args.manifest), resolve(args.processed_dir)
    s3 = CONFIG["s3"]
    if args.s3:
        storage.download(storage.key(s3["manifest_prefix"], manifest_path.name), manifest_path, skip_existing=False)
    df = filter_manifest(pd.read_parquet(manifest_path), args.tags)

    for name in sorted(df["processed_file"].unique()):
        path = processed_dir / name
        if not path.exists():
            if not args.s3:
                raise FileNotFoundError(f"{path} missing. Run preprocess.py or pass --s3.")
            storage.download(storage.key(s3["processed_prefix"], name), path)

    train_df, val_df = split_train_val(df, CONFIG["training"]["train_split"])
    for label, part in (("train", train_df), ("val", val_df)):
        mix = (part["direction"].value_counts(normalize=True) * 100).round(1).to_dict()
        print(f"[DATA] {label}: windows={len(part)} files={part['processed_file'].nunique()} directions%={mix}")
    return (
        make_dataset(to_ranges(train_df), processed_dir, args.batch_size, shuffle=True),
        make_dataset(to_ranges(val_df), processed_dir, args.batch_size, shuffle=False).repeat(),
        len(train_df),
    )


def init_wandb(args, num_train_windows):
    import wandb

    mode = "online" if env("WANDB_API_KEY") else "disabled"
    if mode == "disabled":
        print("[WANDB] WANDB_API_KEY not set; logging to stdout only")
    return wandb.init(
        project=CONFIG["wandb"]["project"],
        entity=env("WANDB_ENTITY"),
        name=args.run_name,
        mode=mode,
        config={**vars(args), **CONFIG["training"], **CONFIG["data"], "stack_size": STACK_SIZE,
                "train_windows": num_train_windows},
    )


def losses(targets, preds):
    huber, mse = tf.keras.losses.Huber(), tf.keras.losses.MeanSquaredError()
    throttle = huber(targets[:, 0:1], preds[:, 0:1])
    brake = huber(targets[:, 1:2], preds[:, 1:2])
    steer = mse(targets[:, 2:3], preds[:, 2:3])
    return {"loss": throttle + brake + steer, "throttle_loss": throttle, "brake_loss": brake, "steer_loss": steer}


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(CONFIG["training"]["seed"])
    train_ds, val_ds, num_train_windows = load_data(args)
    run = init_wandb(args, num_train_windows)

    if args.resume:
        model = load_policy(args.resume)
    else:
        model = build_policy(STACK_SIZE, VISION_SHAPE[:2], LIDAR_SHAPE, TELEMETRY_SHAPE, WAYPOINTS_SHAPE)
    assert model.output_shape[-1:] == CONTROLS_SHAPE
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            out = losses(targets, preds)
        grads = tape.gradient(out["loss"], model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return out, preds

    @tf.function
    def val_step(inputs, targets):
        preds = model(inputs, training=False)
        return losses(targets, preds), preds

    def evaluate(val_iter):
        sums, dir_loss, dir_count = {}, {d: 0.0 for d in DIRECTION_NAMES}, {d: 0 for d in DIRECTION_NAMES}
        for _ in range(args.val_steps):
            inputs, targets, directions = next(val_iter)
            out, preds = val_step(inputs, targets)
            t, p = targets.numpy(), preds.numpy()
            values = {k: float(v) for k, v in out.items()}
            values.update(batch_metrics(t, p))
            for k, v in values.items():
                sums[k] = sums.get(k, 0.0) + v
            for d, (loss, count) in direction_losses(t, p, directions.numpy()).items():
                dir_loss[d] += loss * count
                dir_count[d] += count
        metrics = {k: v / args.val_steps for k, v in sums.items()}
        for d in DIRECTION_NAMES:
            metrics[f"direction_loss/{d}"] = dir_loss[d] / dir_count[d] if dir_count[d] else 0.0
        return metrics

    ckpt_dir = resolve(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_iter = iter(val_ds)
    step, frames_seen, start_time = 0, 0, time.time()
    batches_per_epoch = max(1, -(-num_train_windows // args.batch_size))

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        for batch_idx, (inputs, targets, directions) in enumerate(train_ds, start=1):
            out, preds = train_step(inputs, targets)
            step += 1
            frames_seen += int(targets.shape[0])

            if step == 1 or step % args.log_every == 0:
                val = evaluate(val_iter)
                t, p = targets.numpy(), preds.numpy()
                train = {k: float(v) for k, v in out.items()}
                train.update(batch_metrics(t, p))
                for d, (loss, _) in direction_losses(t, p, directions.numpy()).items():
                    train[f"direction_loss/{d}"] = loss
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{batches_per_epoch} step={step} "
                    f"loss={train['loss']:.5f} val_loss={val['loss']:.5f} "
                    f"val_mae(t,b,s)={val['mae/throttle']:.4f},{val['mae/brake']:.4f},{val['mae/steer']:.4f} "
                    f"val_turn_f1={val.get('turn/f1', 0.0):.3f}"
                )
                payload = {f"train/{k}": v for k, v in train.items()}
                payload.update({f"val/{k}": v for k, v in val.items()})
                payload.update({"progress/epoch": epoch, "progress/frames_seen": frames_seen,
                                "time/elapsed_sec": time.time() - start_time})
                run.log(payload, step=step)

        if step == 0:
            raise RuntimeError("Training dataset produced zero batches. Check the manifest and processed files.")
        print(f"epoch={epoch}/{args.epochs} done in {time.time() - epoch_start:.1f}s")
        run.log({"time/epoch_sec": time.time() - epoch_start, "progress/epoch_completed": epoch}, step=step)

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            path = ckpt_dir / f"policy_epoch_{epoch:03d}.keras"
            model.save(path)
            print(f"[CKPT] {path}")
            if args.s3:
                storage.upload(path, storage.key(CONFIG["s3"]["checkpoint_prefix"], path.name))

    run.finish()


if __name__ == "__main__":
    main()
