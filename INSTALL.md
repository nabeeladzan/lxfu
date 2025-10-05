# LXFU Installation Guide

## File Hierarchy

LXFU follows standard Linux filesystem hierarchy:

```
/usr/bin/lxfu                    # Main executable
/usr/share/lxfu/dino.pt          # DINOv3 model file (read-only)
/etc/lxfu/lxfu.conf              # System-wide configuration
~/.lxfu/                         # User database directory
  ├── lxfu_faces.index           # FAISS index
  └── lxfu_metadata.db/          # LMDB metadata
```

## Build from Source

### Dependencies

```bash
# Arch Linux
sudo pacman -S opencv openblas lapack cmake base-devel

# Ubuntu/Debian
sudo apt install libopencv-dev libopenblas-dev liblapack-dev cmake build-essential

# Fedora
sudo dnf install opencv-devel openblas-devel lapack-devel cmake gcc-c++
```

### Build Steps

```bash
# 1. Download LibTorch (if not already present)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
unzip libtorch-shared-with-deps-2.8.0+cpu.zip

# 2. Build LXFU
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. Install system-wide (requires root)
sudo make install
```

## Installation Steps

### 1. Install the Executable

```bash
sudo install -Dm755 build/bin/lxfu /usr/local/bin/lxfu
```

### 2. Install the Model File

Place the DINOv3 model in the shared data directory:

```bash
sudo mkdir -p /usr/share/lxfu
sudo install -Dm644 dino.pt /usr/share/lxfu/dino.pt
```

### 3. Install Configuration

```bash
sudo mkdir -p /etc/lxfu
sudo install -Dm644 lxfu.conf /etc/lxfu/lxfu.conf
```

Edit `/etc/lxfu/lxfu.conf` if needed:

```bash
sudo nano /etc/lxfu/lxfu.conf
```

### 4. Initialize User Database

The database directory is automatically created on first run:

```bash
lxfu enroll /dev/video0 john
```

This creates `~/.lxfu/` with the FAISS index and LMDB database.

## Usage Modes

### Headless Mode (Default)

By default, LXFU operates without any GUI, perfect for servers, SSH sessions, or scripting:

```bash
# Immediate capture from webcam
lxfu enroll /dev/video0 john

# Query from webcam
lxfu query /dev/video0
```

### Preview Mode (Optional)

Use the `--preview` flag to show a window for visual confirmation:

```bash
# Show preview window, press SPACE to capture, ESC to cancel
lxfu --preview enroll /dev/video0 alice

# Preview before query
lxfu --preview query /dev/video0
```

**Preview Controls:**

- `SPACE` - Capture the current frame
- `ESC` - Cancel operation

**Headless Systems:**
The `--preview` flag will automatically fall back to instant capture mode if:

- No display server is detected (`$DISPLAY` and `$WAYLAND_DISPLAY` unset)
- Window creation fails
- Display operation fails mid-capture

This makes it safe to use `--preview` in scripts that might run on both desktop and server environments.

## Configuration

### Configuration File Locations

LXFU loads configuration in this priority order:

1. `/etc/lxfu/lxfu.conf` - System-wide config
2. `./lxfu.conf` - Local config (for development)

### Configuration Options

```conf
# Path to DINOv3 model file
model_path=/usr/share/lxfu/dino.pt

# Database storage directory
db_path=~/.lxfu

# Default camera device
default_device=/dev/video0
```

### Command-Line Overrides

Command-line arguments always override configuration:

```bash
# Use specific camera device
lxfu enroll /dev/video2 alice

# Use image file instead of camera
lxfu query face.jpg
```

## Uninstallation

```bash
# Remove executable
sudo rm /usr/bin/lxfu

# Remove model and configuration
sudo rm -rf /usr/share/lxfu
sudo rm -rf /etc/lxfu

# Remove user databases (optional)
rm -rf ~/.lxfu
```

## Development Setup

For development, you can run LXFU without installing:

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

## Troubleshooting

### Model file not found

```
Error: Could not load TorchScript model from /usr/share/lxfu/dino.pt
```

**Solution:** Ensure `dino.pt` is installed: `ls -lh /usr/share/lxfu/dino.pt`

### Database permission errors

```
Error: Cannot open LMDB database
```

**Solution:** Check directory exists and is writable: `ls -ld ~/.lxfu`

### Configuration not loading

Add debug output to see which config file is used. The program prints:

```
Loaded config from /etc/lxfu/lxfu.conf
```

### Camera device not found

```
Error: Could not open device: /dev/video0
```

**Solution:** List available cameras: `ls -l /dev/video*`
Then specify correct device: `lxfu enroll /dev/video2 name`
