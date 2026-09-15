# Claim Audit

Every claim in `paper.md` is listed here with its evidence class, following the audit protocol used for the report:

- **M**: quantitatively measured, with a recoverable artifact (training logs, manifest, checkpoint, code)
- **Q**: qualitatively observed (video, screenshot, author observation), no systematic measurement
- **I**: inferred interpretation of M/Q evidence
- **A**: confirmed by the author (first-hand knowledge of the experiment), not independently recoverable from artifacts
- **U**: unsupported or unverified: needs more evidence before it can be stated as fact

Sources: **Logs** = full metric history of training run `barzksni` in [`results/barzksni/`](../results/barzksni) (`history.csv`, `summary.json`), plotted in [`results/training_curves.ipynb`](../results/training_curves.ipynb). **Manifest** = the window manifest from the original `drift` data pipeline, restricted to recordings dated 2026-05-09/10. **Ckpt** = `artifacts/policy.keras` metadata and graph. **Git** = commit history of the original research repository.

| # | Claim | Class | Evidence |
|---|---|---|---|
| 1 | The released policy is the epoch-10 checkpoint of training run `barzksni` | M | Ckpt `date_saved` 2026-05-10 22:19:15 UTC; run created 22:00:32 UTC; Logs show epoch 10 completing at 1,115 s (22:19:07 UTC); checkpointing every 10 epochs |
| 2 | Training set: 236,882 five-frame windows from 448 recordings | M | Logs `sampling/manifest_sampled/windows` = 236,882; Manifest rows dated 05-09/10 = 236,882 across 448 files |
| 3 | Direction mix 89.47% straight / 5.85% left / 4.67% right | M | Logs sampling ratios and Manifest agree to 4 significant figures |
| 4 | ≈3.3 h of 20 Hz driving | M (derived) | 236,882 windows / 20 Hz; each file has at most 4 extra frames not counted as windows |
| 5 | Most recordings are 30 s systematic route captures | M + I | Median 595 windows per file (600 frames = 30 s at 20 Hz = `collect.py` defaults); route tool committed 2026-05-05, before recording |
| 6 | Train/val = 165,777 / 71,108 windows; 53 epochs; 21,750 steps; 8.88 M windows; 1.64 h | M | Logs |
| 7 | Epoch-10 validation: loss 1.53e-3; MAE throttle 0.0170, brake 0.0165, steer 0.0133; turn-F1 0.78 | M | Mean of all validation evaluations during epoch 10 (Logs; notebook Table 1) |
| 8 | Per-direction val loss at epoch 10: straight 0.84e-3, left 0.98e-3, right 1.57e-3 | M | Logs |
| 9 | Late training (epochs 50–53): val loss 0.70e-3, steer MAE 0.0059, turn-F1 0.82 | M | Logs |
| 9a | Training briefly destabilized right after epoch 10 (val loss spike to 5.7e-2, turn-F1 to 0) and recovered by epochs 20–22 | M | Logs; notebook loss and turn plots |
| 10 | Validation windows come from the same recordings as training (temporal hold-out, not route/map hold-out) | M | `split_train_val` in code; Logs split counts |
| 11 | Per-batch R² is strongly negative and not interpretable | M + I | Logs; non-shuffled validation batches have near-zero target variance |
| 12 | 1,358,211 parameters; exact layer dimensions | M | Ckpt |
| 13 | Positional embedding is a fixed random constant, not learned | M | Ckpt `config.json` has no Embedding layer; constant tensor stored as an input to `add` |
| 14 | Released policy (Keras) and ONNX export agree to < 4e-7 | M | `export_onnx.py` parity check |
| 15 | Recovery-route data (4,350 windows) is **not** in the training set | M | Manifest: `recovery` tag appears only on 2026-05-17 recordings |
| 16 | DAgger-mode data is not in the training set | A | Author-confirmed |
| 17 | Training maps are Town01 and Town02 | A + M (partial) | Author-confirmed; manifest tags cover 36% of windows (49,072 map01/map02, 37,174 map01) and show no other map |
| 18 | The policy drives autoregressively (alone on the road, no intervention) for hours on training and held-out routes without collisions | Q + A | Author-run sessions; demo video, GIF; not logged as a benchmark |
| 19 | The policy drives in a CARLA town absent from its demonstrations (zero-shot transfer to an unseen town) | Q + A | `curved_road_carla.png` and video: driving on a map other than Town01/Town02 (author-confirmed); map name intentionally not specified; not yet systematic |
| 20 | The policy sometimes recovers from off-road or unusual states | Q | Author observation; quantifiable with the offset-initialization protocol (paper §7.4), not yet run |
| 21 | Recovery is not explained by recovery demonstrations | I | Supported by 15 and 16; the only remaining source of off-nominal training states is start-pose jitter (±0.25 m, ±3°) |
| 22 | ≈7 minutes of demonstrations was enough for recognizable Town01 driving | U | Author-reported; no dataset/run in this artifact |
| 23 | ≈1 in 10 early rollouts ended in collision | U | Author-reported; no rollout log |
| 24 | The policy receives privileged information at inference | M | Waypoints come from the CARLA HD-map lane graph and the ground-truth ego pose; debug waypoints appear in RGB frames |
| 25 | The policy "generalizes robustly" / "has learned a general recovery strategy" | U: **not claimed** | Would need systematic multi-map, multi-seed rollouts |
| 26 | The architecture is novel | **Not claimed** | |
