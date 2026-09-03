# Swallow Detection Data Viewer

Browser-based dashboard for visualizing swallow detection data — phase displacement, Doppler spectrogram, IMU2 acceleration, and model predictions, all time-aligned.

## Quick Start

### 1. Export session data

From the project root:

```bash
# Export one person
python docs/export_session_data.py --root data/001

# Export specific people
python docs/export_session_data.py --root data/001 data/002 --people 001 002

# Export all sessions
python docs/export_session_data.py --all
```

Exported JSON files go to `docs/data/{person}/{session}.json`, plus a `docs/data/manifest.json` index.

### 2. Launch local server

```bash
python3 -m http.server 8765 --directory docs
```

Open http://localhost:8765 in a browser.

### 3. GitHub Pages

Push the `data-visualization` branch, then in repo Settings → Pages, set source to the `data-visualization` branch, `/docs` folder. The site will be available at `https://<user>.github.io/the_necklace/`.

> **Note:** Each session JSON is ~2.5 MB. For many sessions, consider exporting only the people you need rather than `--all`.

## Dashboard Controls

| Control | Description |
|---------|-------------|
| **Person** dropdown | Select subject folder (e.g. `001`) |
| **Session** dropdown | Select recording session (e.g. `15ml-1`) |
| **RX chips** | Switch between RX0 / RX1 / RX2 for phase plot |
| **Bin chips** | Switch range bin (4–8) for phase plot |
| **Zoom/Pan** | Drag on any plot — all 4 plots sync their x-axis |

## Plot Rows

1. **Swallow Detection** — Feature vector norm (blue) and ground-truth labels (red). Blue shaded regions mark labeled swallow events.
2. **Phase Displacement** — Unwrapped phase converted to mm, per-RX per-bin. Selectable via chips above the plot.
3. **Doppler Spectrogram** — Short-time FFT of the complex range bin signal (RX0). Frequency axis ±50 Hz.
4. **IMU2 Acceleration** — ax (red), ay (green), az (blue), magnitude (orange dashed). Units in g.

## File Structure

```
docs/
├── README.md
├── index.html                  # Dashboard page
├── export_session_data.py      # Export script
└── data/
    ├── manifest.json           # {person: [session, ...]}
    ├── 001/
    │   ├── 15ml-1.json
    │   ├── 15ml-2.json
    │   └── ...
    └── ...
```

## Export Script Options

```
usage: export_session_data.py [-h] [--root ROOT [ROOT ...]] [--all]
                               [--out OUT] [--people PEOPLE [PEOPLE ...]]

  --root     Root dirs containing person/session folders
  --all      Export all sessions under data/
  --out      Output directory (default: docs/data)
  --people   Only export these people
```

Requirements: `numpy`, `pandas`, `scipy`, and the project's `src/radar_swallow_detector_14.py`.
