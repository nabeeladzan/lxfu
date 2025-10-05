# LXFU Quick Reference

## Command Syntax

```bash
lxfu [--preview] <command> <arguments>
```

## Commands

### Enroll

Register a face with a name:

```bash
lxfu enroll <source> <name>
lxfu --preview enroll <source> <name>
```

**Camera Enrollment (Multi-Frame):**

When using a camera device, LXFU captures multiple frames over 10 seconds for improved accuracy:

```bash
lxfu enroll --device /dev/video0 --name alice
lxfu --preview enroll --device /dev/video0 --name alice
```

- Captures continuously for 10 seconds with countdown
- Filters out frames without detected faces automatically
- Stores all valid frames as separate embeddings
- Instructions guide you to make slight head movements

**Image File Enrollment (Single Frame):**

```bash
lxfu enroll --file face.jpg --name bob
lxfu enroll face.jpg bob  # Legacy syntax
```

### Query

Identify a face:

```bash
lxfu query <source>
lxfu --preview query <source>
```

**Examples:**

```bash
lxfu query /dev/video0               # Webcam (instant)
lxfu query unknown.jpg               # Image file
lxfu --preview query /dev/video0     # With preview window
```

## Options

| Flag        | Description                                                                                                                                    |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `--preview` | Show GUI window for visual confirmation (press SPACE to capture, ESC to cancel). Automatically falls back to instant mode on headless systems. |

## Configuration

### Config File Locations (Priority Order)

1. `/etc/lxfu/lxfu.conf` - System-wide
2. `./lxfu.conf` - Local development

### Config File Format

```conf
# Model location
model_path=/usr/share/lxfu/dino.pt

# Database directory
db_path=~/.lxfu

# Default camera device
default_device=/dev/video0

```

## File Locations

| Path                        | Description   |
| --------------------------- | ------------- |
| `/usr/bin/lxfu`             | Executable    |
| `/usr/share/lxfu/dino.pt`   | DINOv2 model  |
| `/etc/lxfu/lxfu.conf`       | System config |
| `~/.lxfu/lxfu_faces.index`  | FAISS index   |
| `~/.lxfu/lxfu_metadata.db/` | LMDB metadata |

## Preview Mode Controls

When using `--preview` flag:

| Key     | Action                |
| ------- | --------------------- |
| `SPACE` | Capture current frame |
| `ESC`   | Cancel operation      |

## Output Examples

### Successful Enrollment (Camera - Multi-Frame)

```
╔════════════════════════════════════════════════════╗
║  ENROLLMENT - Multi-Frame Capture Mode            ║
╚════════════════════════════════════════════════════╝

Instructions:
  • Look at the camera and stay centered
  • VERY SLIGHTLY move and adjust your head
  • Try small turns left/right and slight up/down
  • Keep your face visible at all times

Capturing frames for 10 seconds...

Warming up camera...

Starting capture...
⏱  10 seconds remaining... (captured 0 valid frames)
⏱  9 seconds remaining... (captured 12 valid frames)
⏱  8 seconds remaining... (captured 23 valid frames)
...
⏱  1 seconds remaining... (captured 89 valid frames)

✓ Capture complete!
  Total frames processed: 98
  Frames with detected faces: 92
  Detection rate: 93.9%

Extracting embeddings from 92 frame(s)...
  Processing frame 1/92...
  Processing frame 10/92...
  Processing frame 20/92...
  ...
  Processing frame 92/92...

╔════════════════════════════════════════════════════╗
║  ✓ ENROLLMENT SUCCESSFUL!                          ║
╚════════════════════════════════════════════════════╝

  Profile: alice
  Embedding dimensions: 384
  New samples added: 92
  Total samples for profile: 92
  Total profiles in database: 1
```

### Successful Enrollment (Image File)

```
Loaded config from /etc/lxfu/lxfu.conf
Loading image from file...
Image loaded: 640x480
✓ Face detected at (142, 98) size 256x256
✓ Cropped to face region: 307x307 (from 640x480)

Extracting embeddings from 1 frame(s)...
  Processing frame 1/1...

╔════════════════════════════════════════════════════╗
║  ✓ ENROLLMENT SUCCESSFUL!                          ║
╚════════════════════════════════════════════════════╝

  Profile: bob
  Embedding dimensions: 384
  New samples added: 1
  Total samples for profile: 1
  Total profiles in database: 2
```

### Successful Query

```
Loaded config from /etc/lxfu/lxfu.conf
Loading/capturing face...
Image loaded: 640x480
Extracting face embedding...
Searching for similar faces...

✓ Face recognized!
  Name: alice
  Similarity: 94.23%
  Face ID: 0
```

### No Match Found

```
⚠ No match found
```

### No Faces Enrolled

```
⚠ No faces enrolled yet. Use 'enroll' command first.
```

## Common Workflows

### First Time Setup

```bash
# Install system-wide
sudo make install

# Enroll first face
lxfu enroll /dev/video0 alice
```

### Daily Usage

```bash
# Quick query (headless)
lxfu query /dev/video0

# With visual confirmation
lxfu --preview query /dev/video0
```

### Development Mode

```bash
# Create local config
echo "model_path=./dino.pt" > lxfu.conf
echo "db_path=./data" >> lxfu.conf

# Build and test
mkdir build && cd build
cmake .. && make
./bin/lxfu enroll /dev/video0 test
```

### Batch Enrollment from Files

```bash
# Enroll multiple people
for image in faces/*.jpg; do
    name=$(basename "$image" .jpg)
    lxfu enroll "$image" "$name"
done
```

### Remote Usage (SSH)

```bash
# Headless mode works perfectly over SSH
ssh user@server 'lxfu query /dev/video0'

# Or with image files
scp face.jpg server:/tmp/
ssh user@server 'lxfu query /tmp/face.jpg'
```

## Troubleshooting

### Model not found

```bash
# Check model location
ls -lh /usr/share/lxfu/dino.pt

# Or use local model
echo "model_path=./dino.pt" > lxfu.conf
```

### Camera not found

```bash
# List available cameras
ls -l /dev/video*

# Use specific camera
lxfu enroll /dev/video2 alice
```

### Database not found

```bash
# Check database directory
ls -la ~/.lxfu/

# Or create manually
mkdir -p ~/.lxfu
```

### Preview window not showing

```bash
# Check if X11 is available
echo $DISPLAY

# If over SSH, enable X11 forwarding
ssh -X user@server

# Note: --preview automatically falls back to instant mode on headless systems
lxfu --preview enroll /dev/video0 alice
# Output: ⚠ Warning: --preview requested but no display detected (headless system)
#         ⚠ Falling back to instant capture mode...
#         ✓ Frame captured (instant mode)
```

## Performance Tips

- **Headless mode**: Faster startup, no GUI overhead
- **Preview mode**: Use when you need visual confirmation
- **FAISS search**: Sub-millisecond for thousands of faces
- **Database**: Stored in home directory, automatically created

## Integration Examples

### Shell Script

```bash
#!/bin/bash
result=$(lxfu query /dev/video0 2>&1)
if echo "$result" | grep -q "✓ Face recognized"; then
    name=$(echo "$result" | grep "Name:" | cut -d':' -f2 | xargs)
    echo "Welcome, $name!"
else
    echo "Unknown person"
fi
```

### Python

```python
import subprocess
import re

result = subprocess.run(['lxfu', 'query', '/dev/video0'],
                       capture_output=True, text=True)

if '✓ Face recognized' in result.stdout:
    match = re.search(r'Name: (\w+)', result.stdout)
    if match:
        print(f"Welcome, {match.group(1)}!")
else:
    print("Unknown person")
```

### REST API Wrapper

```python
from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/enroll', methods=['POST'])
def enroll():
    name = request.json['name']
    image_path = request.json['image_path']

    result = subprocess.run(['lxfu', 'enroll', image_path, name],
                           capture_output=True, text=True)

    return jsonify({'success': '✓' in result.stdout})

@app.route('/query', methods=['POST'])
def query():
    image_path = request.json['image_path']

    result = subprocess.run(['lxfu', 'query', image_path],
                           capture_output=True, text=True)

    if '✓ Face recognized' in result.stdout:
        name_match = re.search(r'Name: (\w+)', result.stdout)
        sim_match = re.search(r'Similarity: ([\d.]+)%', result.stdout)

        return jsonify({
            'recognized': True,
            'name': name_match.group(1) if name_match else None,
            'similarity': float(sim_match.group(1)) if sim_match else None
        })

    return jsonify({'recognized': False})
```
