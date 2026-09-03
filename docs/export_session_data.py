#!/usr/bin/env python3
"""Export per-session data for the web dashboard.

For each session, computes and saves:
  - Model swallow probability (per window)
  - Phase displacement (multiple bins, multiple RX)
  - Doppler spectrogram
  - IMU2 signal (ax, ay, az)
  - Labeled swallow events

Output: docs/data/{person}/{session}.json

Usage:
  python docs/export_session_data.py --root data/001 data/002
  python docs/export_session_data.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.radar_swallow_detector_14 import (
    load_session,
    extract_range_profile,
    weighted_bin_fusion,
    compute_displacement_signal,
    make_windows,
    extract_features_handcrafted,
    extract_phase_features,
    extract_cross_rx_features,
    extract_cross_rx_phase_features,
    build_temporal_context,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import re

WAVELENGTH = 3e8 / 60.75e9
BIN_INDICES = [4, 5, 6, 7, 8]
WINDOW_MS = 1000.0
OVERLAP = 0.5

# downsample targets for the web
RADAR_MAX_POINTS = 3000
IMU_MAX_POINTS = 5000
DOPPLER_MAX_TIME_BINS = 800


def downsample(t, y, max_points):
    if len(t) <= max_points:
        return t, y
    step = max(1, len(t) // max_points)
    return t[::step], y[::step]


def compute_doppler(range_fft_complex, radar_t, bin_idx, fs):
    sig = range_fft_complex[:, bin_idx]
    sig = sig - np.mean(sig)

    n = len(sig)
    window_len = min(512, max(64, int(2 ** np.floor(np.log2(n / 4)))))
    window_len = min(window_len, n)
    hop_len = max(1, window_len // 4)
    nfft = int(2 ** np.ceil(np.log2(window_len)))
    num_frames = (n - window_len) // hop_len + 1
    if num_frames < 1:
        return None, None, None

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_len) / max(window_len - 1, 1))
    dop_f = (np.arange(-nfft // 2, nfft // 2)) * fs / nfft
    dop_t = np.zeros(num_frames)
    dop_db = np.full((nfft, num_frames), np.nan)

    for fi in range(num_frames):
        s = fi * hop_len
        e = s + window_len
        segment = sig[s:e]
        if np.any(np.isnan(segment)):
            dop_t[fi] = radar_t[min((s + e) // 2, len(radar_t) - 1)]
            continue
        segment = segment * window
        spectrum = np.fft.fftshift(np.fft.fft(segment, nfft))
        dop_db[:, fi] = 20 * np.log10(np.abs(spectrum) + 1e-12)
        dop_t[fi] = radar_t[min((s + e) // 2, len(radar_t) - 1)]

    # downsample time axis if too large
    if num_frames > DOPPLER_MAX_TIME_BINS:
        step = max(1, num_frames // DOPPLER_MAX_TIME_BINS)
        dop_t = dop_t[::step]
        dop_db = dop_db[:, ::step]

    # trim frequency to ±50 Hz
    freq_mask = (dop_f >= -50) & (dop_f <= 50)
    dop_f = dop_f[freq_mask]
    dop_db = dop_db[freq_mask, :]

    return dop_t, dop_f, dop_db


def load_imu2(session_dir):
    csv_candidates = list(session_dir.glob("accel2_*.csv"))
    if not csv_candidates:
        return None, None
    df = pd.read_csv(csv_candidates[0])
    ts_us = df["timestamp_us"].values

    meta_path = session_dir / "accel2_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        host0 = meta["first_packet_host_time"]
        mcu0 = meta["first_timestamp_us"]
        imu_times = host0 + (ts_us - mcu0) / 1e6
    else:
        start_time = float(csv_candidates[0].stem.split("_", 1)[1])
        imu_times = start_time + (ts_us - ts_us[0]) / 1e6

    accel = df[["ax_g", "ay_g", "az_g"]].values.astype(np.float64)
    return accel, imu_times


def run_loso_predictions(session_dirs):
    """Run Leave-One-Session-Out CV and return per-session predictions.

    Returns dict: { session_dir_str: { "predictions": [0/1, ...], "probabilities": [float, ...] } }
    """
    print("\n=== Running LOSO-CV for predictions ===")

    all_feats_list = []
    all_labels_list = []
    all_session_ids = []
    session_n_windows = {}

    for sdir in session_dirs:
        sdir = Path(sdir)
        session_id = str(sdir)
        try:
            frames, frame_times, labels_df, meta = load_session(sdir)
        except FileNotFoundError:
            continue

        fp = meta.get("frame_period_s", 0.001)
        fs = 1.0 / fp
        n_rx = frames.shape[1]

        rx_mag_sigs = []
        rx_disp_sigs = []
        for rx in range(min(n_rx, 3)):
            rmag, rfft = extract_range_profile(frames, rx_idx=rx)
            mag_sig = weighted_bin_fusion(rmag, rfft, BIN_INDICES, WAVELENGTH)
            disp_sig = compute_displacement_signal(rfft, BIN_INDICES, WAVELENGTH)
            rx_mag_sigs.append(mag_sig)
            rx_disp_sigs.append(disp_sig)
        while len(rx_mag_sigs) < 3:
            rx_mag_sigs.append(rx_mag_sigs[0])
            rx_disp_sigs.append(rx_disp_sigs[0])

        wf = max(4, int(round(WINDOW_MS / 1000.0 * fs)))
        hf = max(1, int(round(wf * (1 - OVERLAP))))

        wins = [make_windows(rx_mag_sigs[i], frame_times, labels_df,
                             window_frames=wf, hop_frames=hf) for i in range(3)]
        dwins = [make_windows(rx_disp_sigs[i], frame_times, labels_df,
                              window_frames=wf, hop_frames=hf) for i in range(3)]

        n_win = len(wins[0][0])
        for i in range(n_win):
            w0, w1, w2 = wins[0][0][i], wins[1][0][i], wins[2][0][i]
            d0, d1, d2 = dwins[0][0][i], dwins[1][0][i], dwins[2][0][i]
            feat = np.concatenate([
                extract_features_handcrafted(w0, fs),
                extract_features_handcrafted(w1, fs),
                extract_features_handcrafted(w2, fs),
                extract_cross_rx_features([w0, w1, w2], fs),
                extract_phase_features(d0, fs),
                extract_phase_features(d1, fs),
                extract_phase_features(d2, fs),
                extract_cross_rx_phase_features([d0, d1, d2]),
            ])
            all_feats_list.append(np.nan_to_num(feat, nan=0.0))
            all_labels_list.append(int(wins[0][1][i]))
            all_session_ids.append(session_id)

        session_n_windows[session_id] = n_win
        print(f"  {sdir.parent.name}/{sdir.name}: {n_win} windows "
              f"({int(sum(wins[0][1]))} pos)")

    if not all_feats_list:
        print("  No windows extracted, skipping predictions")
        return {}

    all_feats = np.array(all_feats_list)
    all_labels = np.array(all_labels_list)
    all_sessions = np.array(all_session_ids)

    all_feats_ctx = build_temporal_context(all_feats, all_sessions, context=2)
    print(f"  Total: {len(all_feats)} windows, "
          f"{all_feats.shape[1]}d -> {all_feats_ctx.shape[1]}d features")

    all_prob = np.zeros(len(all_feats))
    unique_sessions = sorted(set(all_session_ids))

    for fold_i, held_out in enumerate(unique_sessions):
        test_mask = all_sessions == held_out
        train_mask = ~test_mask

        y_train_all = all_labels[train_mask]
        train_pos_idx = np.where(y_train_all == 1)[0]
        train_neg_idx = np.where(y_train_all == 0)[0]
        if len(train_pos_idx) == 0:
            continue

        rng = np.random.default_rng(42)
        n_neg = min(len(train_neg_idx), len(train_pos_idx) * 2)
        neg_sampled = rng.choice(train_neg_idx, size=n_neg, replace=False)
        keep = np.sort(np.concatenate([train_pos_idx, neg_sampled]))

        X_train = all_feats_ctx[train_mask][keep]
        y_train = y_train_all[keep]
        X_test = all_feats_ctx[test_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        all_prob[test_mask] = y_prob

        n_test = int(test_mask.sum())
        sdir_short = Path(held_out)
        print(f"    Fold {fold_i+1}/{len(unique_sessions)}: "
              f"{sdir_short.parent.name}/{sdir_short.name} ({n_test} windows)")

    # triangular smoothing per session
    kernel = np.array([0.25, 0.50, 0.25])
    offset = 0
    for sid in unique_sessions:
        n_win = session_n_windows[sid]
        seg = all_prob[offset:offset + n_win]
        if len(seg) >= 3:
            all_prob[offset:offset + n_win] = np.convolve(seg, kernel, mode='same')
        offset += n_win

    all_pred = (all_prob >= 0.5).astype(int)

    results = {}
    offset = 0
    for sid in unique_sessions:
        n_win = session_n_windows[sid]
        results[sid] = {
            "predictions": all_pred[offset:offset + n_win].tolist(),
            "probabilities": all_prob[offset:offset + n_win].tolist(),
        }
        offset += n_win

    tp = int(((all_pred == 1) & (all_labels == 1)).sum())
    fp = int(((all_pred == 1) & (all_labels == 0)).sum())
    fn = int(((all_pred == 0) & (all_labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    print(f"\n  Overall LOSO: F1={f1:.3f} (P={prec:.3f} R={rec:.3f})")

    return results


def collect_external_predictions(predictions_root, person, session_name):
    """Scan predictions/{model_name}/{person}/{session}.json for all models.

    Each JSON must contain:
      { "t": [...], "labels": [...], "predictions": [...], "probabilities": [...] }

    Returns dict: { model_name: { ... } }
    """
    models = {}
    if not predictions_root.is_dir():
        return models
    for model_dir in sorted(predictions_root.iterdir()):
        if not model_dir.is_dir():
            continue
        pred_file = model_dir / person / f"{session_name}.json"
        if not pred_file.exists():
            continue
        try:
            with open(pred_file) as f:
                pred = json.load(f)
            if "t" in pred and "probabilities" in pred:
                models[model_dir.name] = pred
                print(f"    + model '{model_dir.name}' predictions loaded")
        except Exception as e:
            print(f"    [WARN] {model_dir.name}: {e}")
    return models


def export_session(session_dir, out_dir, reference_epoch=None,
                   predictions=None, predictions_root=None):
    session_dir = Path(session_dir)
    person = session_dir.parent.name
    session_name = session_dir.name

    # extract session type from folder name: "15ml-1" → "15ml", "yjy-dry-1" → "dry"
    type_match = re.search(r'(5ml|15ml|dry|level4-free|level4|level7|water|test)', session_name)
    session_type = type_match.group(1) if type_match else "unknown"

    try:
        frames, frame_times, labels_df, meta = load_session(session_dir)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return False

    fp = meta.get("frame_period_s", 0.001)
    fs = 1.0 / fp
    n_rx = frames.shape[1]

    if reference_epoch is None:
        reference_epoch = frame_times[0]

    # aligned time
    radar_t = frame_times - reference_epoch

    # --- Phase displacement for each RX, each bin ---
    phase_data = {}
    range_fft_all_rx = {}
    for rx in range(min(n_rx, 3)):
        rmag, rfft = extract_range_profile(frames, rx_idx=rx)
        range_fft_all_rx[rx] = rfft
        for bi in BIN_INDICES:
            z = rfft[:, bi]
            z_dynamic = z - z.mean()
            phase = np.unwrap(np.angle(z_dynamic))
            disp_mm = phase * WAVELENGTH / (4 * np.pi) * 1000
            t_ds, d_ds = downsample(radar_t, disp_mm, RADAR_MAX_POINTS)
            phase_data[f"rx{rx}_bin{bi}"] = {
                "t": t_ds.tolist(),
                "disp_mm": d_ds.tolist(),
            }

    # --- Doppler spectrogram (RX0, default bin 7) ---
    dop_t, dop_f, dop_db = compute_doppler(
        range_fft_all_rx[0], radar_t, bin_idx=7, fs=fs
    )
    doppler_data = None
    if dop_t is not None:
        doppler_data = {
            "t": dop_t.tolist(),
            "f": dop_f.tolist(),
            "db": np.nan_to_num(dop_db, nan=-120).tolist(),
        }

    # --- IMU2 ---
    imu2_data = None
    accel2, imu2_times = load_imu2(session_dir)
    if accel2 is not None:
        imu2_t = imu2_times - reference_epoch
        mag = np.sqrt(np.sum(accel2 ** 2, axis=1))
        t_ds, ax_ds = downsample(imu2_t, accel2[:, 0], IMU_MAX_POINTS)
        _, ay_ds = downsample(imu2_t, accel2[:, 1], IMU_MAX_POINTS)
        _, az_ds = downsample(imu2_t, accel2[:, 2], IMU_MAX_POINTS)
        _, mag_ds = downsample(imu2_t, mag, IMU_MAX_POINTS)
        imu2_data = {
            "t": t_ds.tolist(),
            "ax": ax_ds.tolist(),
            "ay": ay_ds.tolist(),
            "az": az_ds.tolist(),
            "mag": mag_ds.tolist(),
        }

    # --- Labels ---
    label_events = []
    if labels_df is not None:
        for _, row in labels_df.iterrows():
            label_events.append({
                "id": int(row["Event_ID"]),
                "start": float(row["Start_Host_Time"]) - reference_epoch,
                "stop": float(row["Stop_Host_Time"]) - reference_epoch,
                "tag": str(row.get("Tag", "swallow")),
            })

    # --- Model predictions (multi-model) ---
    models_data = {}

    # Built-in: 3-RX Phase RF with LOSO-CV
    try:
        rx_mag_sigs = []
        rx_disp_sigs = []
        for rx in range(min(n_rx, 3)):
            rmag, rfft = extract_range_profile(frames, rx_idx=rx)
            mag_sig = weighted_bin_fusion(rmag, rfft, BIN_INDICES, WAVELENGTH)
            disp_sig = compute_displacement_signal(rfft, BIN_INDICES, WAVELENGTH)
            rx_mag_sigs.append(mag_sig)
            rx_disp_sigs.append(disp_sig)
        while len(rx_mag_sigs) < 3:
            rx_mag_sigs.append(rx_mag_sigs[0])
            rx_disp_sigs.append(rx_disp_sigs[0])

        wf = max(4, int(round(WINDOW_MS / 1000.0 * fs)))
        hf = max(1, int(round(wf * (1 - OVERLAP))))

        wins = [
            make_windows(rx_mag_sigs[i], frame_times, labels_df,
                         window_frames=wf, hop_frames=hf)
            for i in range(3)
        ]
        dwins = [
            make_windows(rx_disp_sigs[i], frame_times, labels_df,
                         window_frames=wf, hop_frames=hf)
            for i in range(3)
        ]

        win_times = [(frame_times[s] - reference_epoch,
                       frame_times[min(s + wf - 1, len(frame_times) - 1)] - reference_epoch)
                      for s in range(0, len(frame_times) - wf + 1, hf)]

        feats = []
        for i in range(len(wins[0][0])):
            w0, w1, w2 = wins[0][0][i], wins[1][0][i], wins[2][0][i]
            d0, d1, d2 = dwins[0][0][i], dwins[1][0][i], dwins[2][0][i]
            feat = np.concatenate([
                extract_features_handcrafted(w0, fs),
                extract_features_handcrafted(w1, fs),
                extract_features_handcrafted(w2, fs),
                extract_cross_rx_features([w0, w1, w2], fs),
                extract_phase_features(d0, fs),
                extract_phase_features(d1, fs),
                extract_phase_features(d2, fs),
                extract_cross_rx_phase_features([d0, d1, d2]),
            ])
            feats.append(np.nan_to_num(feat, nan=0.0))

        rf_model = {
            "t": [(s + e) / 2 for s, e in win_times[:len(feats)]],
            "labels": [int(wins[0][1][i]) for i in range(len(feats))],
        }

        sid = str(session_dir)
        if predictions and sid in predictions:
            pred_info = predictions[sid]
            rf_model["predictions"] = pred_info["predictions"]
            rf_model["probabilities"] = pred_info["probabilities"]

        models_data["3rx-phase-rf"] = rf_model

    except Exception as e:
        print(f"  [WARN] Feature extraction failed: {e}")

    # External model predictions from predictions/ directory
    if predictions_root:
        ext_models = collect_external_predictions(
            predictions_root, person, session_name)
        models_data.update(ext_models)

    # --- Assemble output ---
    result = {
        "person": person,
        "session": session_name,
        "session_type": session_type,
        "reference_epoch": reference_epoch,
        "duration_s": float(radar_t[-1] - radar_t[0]),
        "fs": fs,
        "n_rx": min(n_rx, 3),
        "bins": BIN_INDICES,
        "phase": phase_data,
        "doppler": doppler_data,
        "imu2": imu2_data,
        "labels": label_events,
        "models": models_data,
    }

    person_dir = out_dir / person
    person_dir.mkdir(parents=True, exist_ok=True)
    out_path = person_dir / f"{session_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"  Exported: {person}/{session_name} ({size_kb:.0f} KB)")
    return True


def build_manifest(out_dir):
    """Build manifest.json listing all exported sessions."""
    manifest = {}
    for person_dir in sorted(out_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person = person_dir.name
        sessions = []
        for json_file in sorted(person_dir.glob("*.json")):
            sessions.append(json_file.stem)
        if sessions:
            manifest[person] = sessions

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {sum(len(v) for v in manifest.values())} sessions, "
          f"{len(manifest)} people -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Export session data for web dashboard")
    parser.add_argument("--root", nargs="+", type=Path,
                        help="Root dirs containing person/session folders")
    parser.add_argument("--all", action="store_true",
                        help="Export all sessions under data/")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "data",
                        help="Output directory (default: docs/data)")
    parser.add_argument("--people", nargs="+",
                        help="Only export these people")
    parser.add_argument("--predict", action="store_true",
                        help="Run LOSO-CV and include per-session predictions")
    args = parser.parse_args()

    data_root = ROOT / "data"
    if args.all:
        roots = [data_root]
    elif args.root:
        roots = args.root
    else:
        parser.error("Provide --root or --all")

    # discover sessions
    sessions = []
    for root in roots:
        for label_csv in sorted(root.rglob("labeled_swallow.csv")):
            sdir = label_csv.parent
            if (sdir / "frames.bin").exists() and (sdir / "meta.json").exists():
                person = sdir.parent.name
                if args.people and person not in args.people:
                    continue
                sessions.append(sdir)

    print(f"Found {len(sessions)} sessions to export")
    args.out.mkdir(parents=True, exist_ok=True)

    predictions = {}
    if args.predict and len(sessions) >= 2:
        predictions = run_loso_predictions(sessions)

    predictions_root = ROOT / "predictions"

    exported = 0
    for sdir in sessions:
        ok = export_session(sdir, args.out, predictions=predictions,
                            predictions_root=predictions_root)
        if ok:
            exported += 1

    print(f"\nExported {exported}/{len(sessions)} sessions")
    build_manifest(args.out)


if __name__ == "__main__":
    main()
