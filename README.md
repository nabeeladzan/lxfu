# LXFU

### Key Features

- 🚀 **Headless Operation**: No GUI dependencies by default, perfect for servers and embedded systems
- 🖼️ **Optional Preview Mode**: Use `--preview` flag for GUI confirmation when needed (gracefully falls back on headless systems)
- � **Face Detection**: Automatic face detection using Haar cascades for improved accuracy
- �📹 **Webcam & File Support**: Enroll/query from camera devices or image files
- 📊 **Cosine Similarity**: L2-normalized embeddings with inner product search
- 💾 **Persistent Storage**: FAISS index + LMDB for fast retrieval
- ⚙️ **Configuration System**: Standard Linux configuration with `/etc/lxfu/lxfu.conf`
- 📦 **FHS Compliant**: Follows Filesystem Hierarchy Standard
- ⚡ **Fast Search**: Sub-millisecond face matching
- 🎓 **DINOv2 Embeddings**: State-of-the-art self-supervised vision featurese Utility

A high-performance headless face recognition CLI tool using DINOv2, FAISS, and LMDB.

## Overview

LXFU is a production-ready face recognition system designed for Linux environments. It uses Meta's DINOv2 vision transformer for feature extraction, FAISS for efficient similarity search, and LMDB for metadata storage.

## Installation

See [INSTALL.md](INSTALL.md) for complete installation instructions.

### Quick Install

```bash
# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install system-wide
sudo make install
```

### File Locations

```
/usr/local/bin/lxfu              # Executable
/usr/share/lxfu/dino.pt          # Model file
/etc/lxfu/lxfu.conf              # System config
~/.lxfu/                         # User databases
```

## Usage

### Configuration

LXFU reads configuration from:

1. `/etc/lxfu/lxfu.conf` (system-wide)
2. `~/.config/lxfu/lxfu.conf` (user-specific, overrides system)
3. `./lxfu.conf` (development mode)

Command-line arguments always override configuration.

### Enroll a Face

**Headless mode (default):**

```bash
# Uses default camera (/dev/video0) and stores under name "default"
lxfu enroll

# Explicit device/name selection
lxfu enroll --device /dev/video0 --name nabeel

# Image enrollment (legacy positional arguments still supported)
lxfu enroll face.jpg bob
```

**Preview mode (with GUI confirmation):**

```bash
# Show preview window, press SPACE to capture, ESC to cancel
lxfu --preview enroll --device /dev/video0 --name nabeel

# Preview image before processing
lxfu --preview enroll --file photo.jpg --name bob

# Note: On headless systems (no DISPLAY), automatically falls back to instant mode
```

### Query a Face

**Headless mode (default):**

```bash
# Query the default profile using the default device
lxfu query

# Require a specific profile name
lxfu query --device /dev/video0 --name nabeel

# Accept any enrolled profile
lxfu query --device /dev/video0 --all
```

**Preview mode (with GUI confirmation):**

```bash
# Show preview window before query
lxfu --preview query --device /dev/video0 --name nabeel

# Preview image before query
lxfu --preview query --file unknown.jpg --all
```

### Output Example

```
Loaded config from /etc/lxfu/lxfu.conf
Loading/capturing face...
Image loaded: 640x480
Extracting face embedding...
Searching for similar faces...

✓ Face recognized!
  Name: alice (requested)
  Similarity: 94.23%
  Face ID: 0
```

### Preview Mode & Headless Fallback

- `--preview` shows a live window with on-screen instructions; press SPACE to capture the current frame or ESC to cancel.
- When a display server is missing or OpenCV cannot open a window, LXFU logs a warning and continues with an instant capture so scripts do not break.
- The same fallback applies to image previews, making `--preview` safe to use across desktops, servers, SSH sessions, CI jobs, and containers.
- Preview overlays face bounding boxes when the Haar cascade is available, providing immediate feedback before capture.

## Configuration File

Edit `/etc/lxfu/lxfu.conf`:

```conf
# Path to DINOv2 model file
model_path=/usr/share/lxfu/dino.pt

# Database storage directory
db_path=~/.lxfu

# Default camera device
default_device=/dev/video0
```

## Architecture

```
┌─────────────────────────────────────────┐
│  Input: Webcam or Image File           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Haar Cascade Face Detection            │
│  - Detect largest face in image         │
│  - Crop with 20% padding                │
│  - Fall back to full image if no face   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  OpenCV: Image Preprocessing            │
│  - Resize to 224x224                    │
│  - Normalize (ImageNet stats)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  DINOv2: Feature Extraction             │
│  - TorchScript model (LibTorch)         │
│  - Output: 384-dim embedding vector     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  L2 Normalization                       │
│  - Prepare for cosine similarity        │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ FAISS Index  │  │  LMDB Store  │
│ - Embeddings │  │  - ID → Name │
│ - IndexFlatIP│  │  - Metadata  │
└──────────────┘  └──────────────┘
```

## Technical Details

### Face Detection

- Runs OpenCV's Haar cascade to locate faces, selects the largest detection, and adds roughly 20% padding before cropping.
- Falls back to the full frame whenever no face is detected or the cascade file is unavailable, so enrollment/query still work.
- Preview mode overlays green rectangles and padded crop hints whenever the cascade is loaded successfully.

**Pipeline**

1. Convert the captured frame to grayscale and equalize the histogram for stable detections.
2. Execute the Haar cascade and take the largest bounding box as the primary subject.
3. Expand the crop with padding and clamp it to image bounds before preprocessing for DINOv2.

**Haar cascade location hints**

- Searches standard directories such as `/usr/share/opencv4/haarcascades/`, `/usr/local/share/opencv4/haarcascades/`, and project-local copies.
- Install `haarcascade_frontalface_default.xml` via your distribution’s OpenCV packages (e.g., `libopencv-dev`, `opencv-samples`) or place the XML alongside the binary during development.

### Embedding & Search Pipeline

- Frames are resized to 224×224 and normalized with ImageNet statistics before being passed to the TorchScript DINOv2-small model (384-dimensional embeddings).
- Embeddings are L2-normalized and stored in a FAISS `IndexFlatIP`, making inner-product search equivalent to cosine similarity.
- FAISS is accelerated with OpenBLAS; LMDB maintains the ID → name metadata mapping.

### Storage Layout

- Embedding index: `~/.lxfu/lxfu_faces.index` (configurable via `db_path`).
- Metadata store: `~/.lxfu/lxfu_metadata.db/`.
- Model path defaults to `/usr/share/lxfu/dino.pt` but can be overridden in the configuration file.

## Prerequisites

- CMake ≥ 3.18
- C++17 compiler (GCC ≥ 9, Clang ≥ 10)
- LibTorch 2.8.0+ (CPU)
- OpenCV 4.x (with objdetect module for Haar cascades)
- OpenBLAS & LAPACK
- FAISS (included as submodule)
- LMDB (included as submodule)

**Note**: For face detection, you may need to install OpenCV's data files:

```bash
# Arch Linux
sudo pacman -S opencv-samples

# Ubuntu/Debian
sudo apt install libopencv-dev

# The Haar cascade XML files are usually installed with OpenCV
```

## PAM Module

Building the project also produces `pam_lxfu.so`, a PAM module that lets you plug face verification into stacks such as hyperlock or login:

- Install with `sudo cmake --build build --target install` to drop the module in `/usr/lib/security`.
- Add it to a service, e.g. `auth sufficient pam_lxfu.so device=/dev/video0 threshold=0.92`.
- The module runs headless (no preview) and falls back to other PAM entries when the face database is empty or the match is below the threshold.
- Optional module options mirror the CLI: `name=<profile>` to require a specific name (defaults to the PAM user) and `allow_all=true` to accept any enrolled profile.

## Development

For development without system installation:

````
$ ./build/bin/lxfu enroll face.jpg nabeel
Loading DINOv2 model on CPU...
Loading/capturing face...
Image loaded: 800x600
Extracting face embedding...
Embedding extracted: 1024 dimensions
Loading existing FAISS index...
Loaded index with 5 faces
Face added to index with ID: 5
Name 'nabeel' associated with face ID 5
Index saved with 6 faces

✓ Enrollment successful!
  Name: nabeel
  Face ID: 5
## Development

For development without system installation:

```bash
# Create local config
cp lxfu.conf lxfu.conf.local
nano lxfu.conf.local

# Edit to use local paths:
# model_path=./dino.pt
# db_path=./data

# Run from build directory
./build/bin/lxfu enroll /dev/video0 test
````

## Data Storage

- **FAISS Index**: `~/.lxfu/lxfu_faces.index` - stores embedding vectors
- **LMDB Database**: `~/.lxfu/lxfu_metadata.db/` - stores ID → name mappings
- Location is configurable via `db_path` in configuration file

## Technical Details

### Why Cosine Similarity?

For face/image embeddings, **cosine similarity is superior to Euclidean distance** because:

- Focuses on directional similarity (pattern matching)
- Invariant to embedding magnitude
- Standard in computer vision (CLIP, DINOv2, FaceNet)

### FAISS Index Type

Uses `IndexFlatIP` (Inner Product) on L2-normalized vectors:

- Exact search (no approximation)
- Inner product on normalized vectors = cosine similarity
- Similarity scores: 1.0 (identical) to -1.0 (opposite)

### Performance

- **Embedding extraction**: ~50-200ms (CPU-dependent)
- **FAISS search**: <1ms for thousands of faces
- **LMDB lookup**: <0.1ms per query
- **Total**: Real-time performance for face recognition

## Troubleshooting

See [INSTALL.md](INSTALL.md) for common issues:

- Model file not found
- Database permission errors
- Camera device not found
- Configuration loading

## Future Enhancements

- [ ] Multi-face detection and tracking
- [ ] Face quality assessment
- [ ] Batch enrollment from directories
- [ ] REST API server mode
- [ ] GPU acceleration support
- [ ] Advanced FAISS indexes (IVF, PQ) for millions of faces
- [ ] Face management commands (list, delete, update)

## License

See individual component licenses:

- FAISS: MIT
- LMDB: OpenLDAP Public License
- DINOv2: Apache 2.0
