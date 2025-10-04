# LXFU Changelog

## Recent Changes

### Face Detection (NEW!)

- **Automatic face detection**: Uses OpenCV Haar Cascade to detect and crop faces
- **Improved accuracy**: DINOv2 processes only face region, not background
- **Graceful fallback**: Uses full image if no face detected
- **Multiple face handling**: Automatically selects largest (closest) face
- **Configurable padding**: 20% padding around detected face for context
- **Visual feedback**: Face rectangles drawn in preview mode
- **Smart path detection**: Automatically finds Haar cascade XML files
- **No breaking changes**: Works transparently with existing workflows

### Configuration System

- **Configuration file support**: LXFU now reads from `/etc/lxfu/lxfu.conf` (system) or `./lxfu.conf` (development)
- **Standard Linux paths**: Model at `/usr/share/lxfu/dino.pt`, database at `~/.lxfu/`
- **Configurable options**:
  - `model_path` - DINOv2 model location
  - `db_path` - Database directory (FAISS + LMDB)
  - `default_device` - Default camera device
- **Command-line override**: Arguments override configuration file settings

### Preview Mode

- **`--preview` flag**: Optional GUI mode for visual confirmation
- **Headless by default**: No GUI dependencies when running without `--preview`
- **Graceful fallback**: Automatically switches to instant capture if display unavailable
  - Checks `$DISPLAY` and `$WAYLAND_DISPLAY` environment variables
  - Catches OpenCV exceptions during window operations
  - Warns user and continues without preview
  - Safe to use in scripts for mixed environments (desktop + server)
- **Preview features**:
  - Live camera feed with on-screen instructions
  - Press SPACE to capture frame
  - Press ESC to cancel operation
  - Works with both webcam and image files
- **Proper resource cleanup**: Fixed potential segfault issues with VideoCapture

### CLI Enhancements

- Added flag-based argument parsing for `enroll`/`query` (`--device`, `--file`, `--name`).
- Embeddings are now stored directly in LMDB and compared with cosine similarity—FAISS is no longer required.
- Profile management commands:
  - `lxfu list` enumerates stored profiles and vector dimensions.
  - `lxfu delete --name` removes a profile (with optional `--confirm`).
  - `lxfu clear` wipes all stored profiles (with optional `--confirm`).

### Usage Examples

**Headless operation (default):**

```bash
lxfu enroll /dev/video0 alice    # Instant capture
lxfu query /dev/video0           # Instant query
```

**With preview:**

```bash
lxfu --preview enroll /dev/video0 alice   # Show preview, press SPACE
lxfu --preview query /dev/video0          # Show preview before query
```

### Technical Improvements

- **Resource Management**: Proper VideoCapture cleanup prevents segmentation faults
  - `cap.release()` called in all code paths
  - `cv::destroyAllWindows()` + `cv::waitKey(1)` for proper window cleanup
  - Frame cloning to avoid use-after-free issues
- **Error Handling**: Comprehensive exception handling for all operations
- **Configuration Hierarchy**: Follows XDG Base Directory specification

### File Structure

```
/etc/lxfu/lxfu.conf              # System configuration
/usr/share/lxfu/dino.pt          # Model file
/usr/bin/lxfu                    # Executable
~/.lxfu/                         # User databases
  └── embeddings/                # LMDB environment holding profile vectors
```

### Migration Notes

If you have an existing LXFU installation:

1. **Database location changed**: Databases now in `~/.lxfu/` instead of current directory

   - Old (FAISS-based): `./lxfu_faces.index`, `./lxfu_metadata.db/`
   - New: `~/.lxfu/embeddings/` (LMDB environment)
   - Migration: re-enroll faces or write a one-off conversion script if needed

2. **Model location**: Place `dino.pt` in `/usr/share/lxfu/` for system-wide installation

   - Development: Can still use local `dino.pt` with `./lxfu.conf`

3. **Command-line syntax**: `--preview` flag must come before command
   - Old: N/A (no preview mode)
   - New: `lxfu --preview enroll /dev/video0 name`

### Configuration File Format

```conf
# /etc/lxfu/lxfu.conf

# Path to DINOv2 model file
model_path=/usr/share/lxfu/dino.pt

# Database storage directory (~ expands to $HOME)
db_path=~/.lxfu

# Default camera device
default_device=/dev/video0
```

### Development Workflow

For local development without system installation:

```bash
# Create local config
cat > lxfu.conf << EOF
model_path=./dino.pt
db_path=./data
default_device=/dev/video0
EOF

# Build and run
mkdir build && cd build
cmake .. && make -j$(nproc)
./bin/lxfu enroll /dev/video0 test

# Or with preview
./bin/lxfu --preview enroll /dev/video0 test
```

### Future Enhancements

- [ ] Environment variable support (e.g., `LXFU_MODEL_PATH`)
- [ ] `--config` flag to specify custom config file
- [ ] `lxfu list` command to show enrolled faces
- [ ] `lxfu delete <name>` command to remove faces
- [ ] `--threshold` flag to set similarity threshold
- [ ] JSON output mode for scripting
