# binding

CLI tools for analyzing converted microscopy TIFF folders. Each workflow is a small recipe of `binding` commands that share a common input layout and write predictable outputs.

## Install

```bash
uv sync
```

The `binding` entry point is available in the project virtual environment.

## Input layout

Most commands expect data converted from ND2 (or similar) into this folder structure:

```text
dataset/
  Pos0/
    img_channel000_position000_time000000000_z000.tif
    img_channel001_position000_time000000000_z000.tif
    ...
```

Filename pattern: `img_channel{C}_position{P}_time{T}_z{Z}.tif`

Some workflows also use pre-extracted per-cell crops:

```text
dataset/
  roi/
    Pos0/
      index.json
      Roi0.tif
      Roi1.tif
      ...
  bbox/
    Pos0.csv
```

Optional metadata for physical units:

```text
metadata.json   # normalized.pixel_size_um with x, y, z
```

## Workflows

### 1. 3D particle segmentation (watershed)

Use when particles are volumetric blobs that should be segmented, measured, and optionally classified relative to a membrane.

```text
show -> binarize -> spotiflow -> label -> analyze -> analyze-membrane -> plot / show-result
```

```bash
# Inspect one stack
binding show DATA -p 0 -c 0 -t 0 --metadata metadata.json

# Threshold to binary mask (interactive napari contrast, or fixed threshold)
binding binarize DATA -p 0 -c 0 -t 0 -o masks/ --metadata metadata.json
binding binarize DATA -p 0 -c 0 -t 0 -o masks/ --threshold 1200

# Detect watershed seeds (single 3D stack; defaults to smfish_3d for 3D input)
binding spotiflow DATA -p 0 -c 0 -t 0 -o seeds/

# Label connected components from binary + seeds
binding label masks/binary_position000_channel000_time000000000.npy \
  --seeds seeds/spotiflow_position000_channel000_time000000000.csv

# Measure volume, intensity, centroid
binding analyze DATA -p 0 -c 0 -t 0 \
  --mask masks/binary_position000_channel000_time000000000_labeled.npy \
  -o analysis/

# Distance to membrane surface
binding analyze-membrane analysis/analysis_position000_channel000_time000000000.csv \
  --membrane-mask membrane.npy \
  --metadata metadata.json

# Explore results
binding plot analysis/analysis_position000_channel000_time000000000.csv
binding show-result masks/binary_position000_channel000_time000000000_labeled.npy \
  --analysis analysis/analysis_position000_channel000_time000000000.csv \
  --membrane-result analysis/analysis_position000_channel000_time000000000_membrane.csv \
  --metadata metadata.json
```

**Outputs**

| Step | Output |
|---|---|
| `binarize` | `binary_position###_channel###_time#########.npy` |
| `spotiflow` | `spotiflow_position###_channel###_time#########.csv` |
| `label` | `*_labeled.npy` |
| `analyze` | `analysis_position###_channel###_time#########.csv` |
| `analyze-membrane` | `*_membrane.csv` |
| `plot` | `*_histograms.png` |

### 2. Automatic 3D segmentation (no Spotiflow seeds)

Use when Yen thresholding plus z-linked components is enough.

```bash
binding segment DATA -p 0 -c 0 -t 0 -o masks/ --min-z-planes 2
binding analyze DATA -p 0 -c 0 -t 0 \
  --mask masks/binary_position000_channel000_time000000000_labeled.npy \
  -o analysis/
binding show-labeled masks/binary_position000_channel000_time000000000_labeled.npy \
  --metadata metadata.json
```

### 3. ROI mean-intensity time course

Use when the goal is fluorescence intensity in fixed square ROIs over time, not spot counting.

```bash
binding timeseries DATA -p 0 -c 1 \
  --sizes 8,16,32,64,128 \
  --center-y 128 --center-x 128 \
  --time-map time_map.csv \
  -o timeseries/

binding show-timeseries timeseries/timeseries_position000_channel001.csv \
  --use-time-real -o timeseries/timeseries_position000_channel001.png
```

**Outputs**

| Step | Output |
|---|---|
| `timeseries` | `timeseries_position###_channel###.csv` |
| `show-timeseries` | `*_timeseries.png` |

### 4. LNP spot counting per cell (2D Spotiflow)

Use for fluorescently labeled point-like clusters in per-cell ROI crops over a time-lapse. Spotiflow detections are the final result (no watershed).

```text
spotiflow --roi-stacks -> filter-spots -> spot-counts -> plot-lnp
```

```bash
# Detect spots in every cell ROI and timepoint (channel 1 = fluorescence)
binding spotiflow DATA \
  --roi-stacks --all-rois --all-times \
  -p 0 -c 1 \
  --model general --estimate-params \
  -o results/spotiflow

# Filter by intensity, size (FWHM), and detection probability
binding filter-spots results/spotiflow -o results/filtered \
  --min-intensity 4000 \
  --min-fwhm 2.0 --max-fwhm 6.0 \
  --min-probability 0.4

# Count spots per cell over time (cumulative unique spots by default)
binding spot-counts results/filtered -o results/ \
  --time-interval 40 --cumulative

# Three-panel figure: fluorescence, detections, time course
binding plot-lnp DATA results/spot_counts_position000_channel001.csv \
  --filtered-dir results/filtered \
  -o results/fig5_panels.png \
  -c 1 --roi 2 --time 72 --time-unit min
```

**Time axis:** `spot-counts` stores `time_real` in seconds. `plot-lnp` displays minutes by default (`--time-unit min`). For subsampled ND2 data where every 10th frame was kept from a 4 s acquisition, use `--time-interval 40`.

**Outputs**

| Step | Output |
|---|---|
| `spotiflow` | `roi##_time#########.csv` per ROI and time |
| `filter-spots` | `roi##_time#########_filtered.csv` |
| `spot-counts` | `spot_counts_position###_channel###.csv` |
| `plot-lnp` | `fig5_panels.png` (or custom `--output`) |

## Command reference

| Command | Role |
|---|---|
| `show` | View a raw stack in napari |
| `binarize` | Threshold stack to binary `.npy` |
| `spotiflow` | Spot detection (3D seeds or 2D ROI batch) |
| `label` | Seeded watershed on binary mask |
| `segment` | Yen threshold + z-linked labeling |
| `analyze` | Measure labeled particles |
| `analyze-membrane` | Particle distance to membrane |
| `label-membrane` | Solidify membrane from binary mask |
| `plot` | Volume/intensity histograms |
| `show-labeled` | View labeled mask in napari |
| `show-result` | View membrane-distance colormap |
| `timeseries` | ROI mean intensity over time |
| `show-timeseries` | Plot intensity time course |
| `filter-spots` | Filter Spotiflow CSVs by intensity/size |
| `spot-counts` | Per-cell spot counts over time |
| `plot-lnp` | LNP three-panel figure (A/B/C) |

## Choosing a workflow

| Data | Goal | Recipe |
|---|---|---|
| 3D z-stack, blob-like particles | Segment and measure volume | Workflow 1 or 2 |
| 3D particles near a membrane | Inside/outside/surface classification | Workflow 1 + `analyze-membrane` |
| Full field, fixed ROI | Intensity drift/binding signal | Workflow 3 |
| Per-cell ROI crops, point spots | Count clusters over time | Workflow 4 |

## Spotiflow models

| Mode | Default model | Notes |
|---|---|---|
| Single 3D stack (`spotiflow DATA -t 0`) | `smfish_3d` | Seeds for watershed |
| 2D ROI batch (`--roi-stacks`) | `general` | Point detections with optional `--estimate-params` |
| Single 2D plane | `general` | Pass `--model general` explicitly |

Other registered models include `fluo_live`, `hybiss`, and `synth_complex`. Use `--model NAME` to override.

## Example dataset (fig5)

```bash
DATA=/home/jack/data/lisca_review/fig5/20260324_1

binding spotiflow "$DATA" --roi-stacks --all-rois --all-times -c 1 \
  --model general --estimate-params -o "$DATA/results/spotiflow"

binding filter-spots "$DATA/results/spotiflow" -o "$DATA/results/filtered"
binding spot-counts "$DATA/results/filtered" -o "$DATA/results/" --time-interval 40
binding plot-lnp "$DATA" "$DATA/results/spot_counts_position000_channel001.csv" \
  --filtered-dir "$DATA/results/filtered" \
  -o "$DATA/results/fig5_panels.png" --time-unit min
```