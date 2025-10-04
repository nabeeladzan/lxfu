# LXFU

### Key Features

- 🚀 **Headless Operation**: No GUI dependencies by default, perfect for servers and embedded systems
- 🖼️ **Optional Preview Mode**: Use `--preview` flag for GUI confirmation when needed (gracefully falls back on headless systems)
- � **Face Detection**: Automatic face detection using Haar cascades for improved accuracy
- �📹 **Webcam & File Support**: Enroll/query from camera devices or image files
- 📊 **Cosine Similarity**: L2-normalized embeddings with inner product search
- 💾 **Persistent Storage**: LMDB database storing normalized embeddings per profile
- ⚙️ **Configuration System**: Standard Linux configuration with `/etc/lxfu/lxfu.conf`
- 📦 **FHS Compliant**: Follows Filesystem Hierarchy Standard
- ⚡ **Fast Search**: Sub-millisecond face matching
- 🎓 **DINOv2 Embeddings**: State-of-the-art self-supervised vision features
- 🔌 **DBus Face Service**: Optional daemon exposing face verification over `dev.nabeeladzan.lxfu` for lock-screen integrations

A high-performance headless face recognition CLI tool using DINOv2 and LMDB.

## Overview

LXFU is a production-ready face recognition system designed for Linux environments. It uses Meta's DINOv2 vision transformer for feature extraction and stores normalized embeddings inside an LMDB database keyed by profile name.

## Installation

See [INSTALL.md](INSTALL.md) for complete installation instructions.

### Quick Install

```bash
# Build and install (installs to /usr with config in /etc)
./build.sh
sudo ./install.sh

# Or manually with CMake
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
sudo make install
```

### File Locations

```
/usr/bin/lxfu                    # Executable
/usr/share/lxfu/dino.pt          # Model file
/etc/lxfu/lxfu.conf              # System config
~/.lxfu/                         # User databases
```

**Note:** To install to `/usr/local` instead:
```bash
INSTALL_PREFIX=/usr/local ./build.sh
sudo INSTALL_PREFIX=/usr/local ./install.sh
```

## Usage

### Configuration

LXFU reads configuration from:

1. `/etc/lxfu/lxfu.conf` (system-wide)
2. `./lxfu.conf` (development mode)

Command-line arguments always override configuration.

### Enroll a Face

LXFU uses an intelligent multi-frame enrollment system for camera-based enrollment to maximize recognition accuracy.

**Camera Enrollment (Multi-Frame Capture):**

When enrolling from a camera, LXFU automatically captures multiple frames over 10 seconds:

```bash
# Multi-frame enrollment with default camera
lxfu enroll --device /dev/video0 --name nabeel

# With live preview window
lxfu --preview enroll --device /dev/video0 --name nabeel
```

**How it works:**
- Captures frames continuously for 10 seconds with a countdown timer
- Automatically filters out frames without detected faces
- Instructs you to make slight head movements for pose variation
- Stores all valid frames as separate embeddings for the profile
- Results in more robust and accurate recognition

**During enrollment you'll see:**
```
╔════════════════════════════════════════════════════╗
║  ENROLLMENT - Multi-Frame Capture Mode            ║
╚════════════════════════════════════════════════════╝

Instructions:
  • Look at the camera and stay centered
  • VERY SLIGHTLY move and adjust your head
  • Try small turns left/right and slight up/down
  • Keep your face visible at all times

⏱  10 seconds remaining... (captured 0 valid frames)
⏱  9 seconds remaining... (captured 8 valid frames)
...
✓ Capture complete!
  Total frames processed: 95
  Frames with detected faces: 87
  Detection rate: 91.6%
```

**Image File Enrollment (Single Frame):**

For image files, traditional single-frame enrollment is used:

```bash
# Enroll from image file
lxfu enroll --file face.jpg --name bob

# Legacy positional syntax still supported
lxfu enroll face.jpg bob

# Preview image before processing
lxfu --preview enroll --file photo.jpg --name bob
```

**Tips for best results:**
- Ensure good lighting conditions
- Keep your face clearly visible and centered
- Make smooth, gentle movements during camera enrollment
- Avoid rapid movements or turning away from the camera
- Multiple enrollment sessions can further improve accuracy

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

### Manage Profiles

```bash
# Show all enrolled profiles and their embedding dimensions
lxfu list

# Delete a stored embedding for a profile (requires confirmation)
lxfu delete --name nabeel --confirm

# Remove every stored profile (asks for confirmation unless --confirm)
lxfu clear --confirm
```

All destructive actions prompt for confirmation unless `--confirm` is supplied.

### Output Example

```
Loaded config from /etc/lxfu/lxfu.conf
Loading/capturing face...
Image loaded: 640x480
Extracting face embedding...
✓ Face recognized!
  Name: alice (requested)
  Similarity: 94.23%
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
│  LMDB Storage                           │
│  - Normalized embedding per profile     │
│  - Cosine similarity via dot product    │
└─────────────────────────────────────────┘
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
- Embeddings are L2-normalized and written directly into LMDB under the profile name.
- Query compares the captured embedding with each stored vector using a cosine (dot product) similarity.

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
- The module compares the captured embedding directly against the stored LMDB profiles using cosine similarity.

## Face Service (DBus)

`lxfu_face_service` exposes face verification over the system bus so compositors can start hands-free auth as soon as the lock screen appears.

- Service name: `dev.nabeeladzan.lxfu`
- Manager object: `/dev/nabeeladzan/lxfu` implementing `dev.nabeeladzan.lxfu.Manager`
  - Method `GetDefaultDevice()` → object path of the active device (`/dev/nabeeladzan/lxfu/Device0`)
- Device object: `/dev/nabeeladzan/lxfu/Device0` implementing `dev.nabeeladzan.lxfu.Device`
  - `Claim()` / `Release()` guard exclusive access to the camera
  - `VerifyStart(mode)` begins a warm-up + multi-frame capture loop (`mode="any"` is currently supported)
  - `VerifyStop()` cancels an in-flight capture
  - Signal `VerificationStatus(status, message)` streams progress (`verify-started`, `verify-match`, `verify-no-face`, `verify-no-match`, `verify-error`, `verify-cancelled`)

Configuration keys (optional) in `lxfu.conf` customise the daemon without recompiling:

```
service_device=/dev/v4l/by-id/<camera>   # defaults to default_device
service_profile=default                  # enforce a specific profile, or combine with service_allow_all=true
service_allow_all=false                  # accept any enrolled profile
service_warmup_delay=1.0                 # seconds to pre-heat IR emitters
service_capture_duration=2.0             # seconds to gather frames per attempt
service_frame_interval=0.1               # delay between frames during capture
service_threshold=0.90                   # cosine similarity threshold (0..1)
```

### Running the daemon

```
cmake --build build --target lxfu_face_service
./build/bin/lxfu_face_service    # foreground test run
```

Installers drop the binary in `/usr/bin/lxfu_face_service`; you can wire it into a systemd service or supervise it from your compositor. Clients should `Claim()`, call `VerifyStart("any")`, react to `VerificationStatus`, and finally `Release()` when finished.

## Development

For development without system installation:

````
$ ./build/bin/lxfu enroll face.jpg nabeel
Loading DINOv2 model on CPU...
Loading/capturing face...
Image loaded: 800x600
Extracting face embedding...
Embedding extracted: 384 dimensions

✓ Enrollment successful!
  Name: nabeel
  Embedding dimensions: 384
  Total profiles: 1

$ ./build/bin/lxfu query face.jpg nabeel
Loading DINOv2 model on CPU...
Loading/capturing face...
Image loaded: 800x600
Extracting face embedding...

✓ Face recognized!
  Name: nabeel (requested)
  Similarity: 98.12%
````

```bash
# Create local config
cp lxfu.conf lxfu.conf.local
nano lxfu.conf.local

# Edit to use local paths:
# model_path=./dino.pt
# db_path=./data

# Run from build directory
./build/bin/lxfu enroll /dev/video0 test
```

## Data Storage

- **LMDB Directory**: `~/.lxfu/embeddings/` - stores normalized embeddings keyed by profile name
- Location is configurable via `db_path` in the configuration file

## Technical Details

### Why Cosine Similarity?

For face/image embeddings, **cosine similarity is superior to Euclidean distance** because:

- Focuses on directional similarity (pattern matching)
- Invariant to embedding magnitude
- Standard in computer vision (CLIP, DINOv2, FaceNet)

### Performance

- **Embedding extraction**: ~50-200ms (CPU-dependent)
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
- [ ] Improve large-scale search options for millions of faces
- [ ] Face management commands (list, delete, update)

## License

See individual component licenses:

- LMDB: OpenLDAP Public License
- DINOv2: Apache 2.0
