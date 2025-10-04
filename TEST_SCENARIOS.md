# LXFU Test Scenarios

## Test Plan for Preview Mode Fallback

### Test 1: Desktop Environment (GUI Available)

**Setup:**

```bash
echo $DISPLAY  # Should show :0 or similar
```

**Test:**

```bash
./build/bin/lxfu --preview enroll /dev/video0 test1
```

**Expected Output:**

```
Loaded config from ./lxfu.conf
Loading/capturing face...
Preview mode: Press SPACE to capture, ESC to cancel...
✓ Frame captured!
Image loaded: 640x480
Extracting face embedding...
✓ Enrollment successful!
```

**Expected Behavior:**

- ✅ Window opens showing live camera feed
- ✅ Instructions visible on screen
- ✅ SPACE captures frame
- ✅ Window closes cleanly
- ✅ No segfault

---

### Test 2: Headless System (No Display)

**Setup:**

```bash
unset DISPLAY
unset WAYLAND_DISPLAY
echo $DISPLAY  # Should be empty
```

**Test:**

```bash
./build/bin/lxfu --preview enroll /dev/video0 test2
```

**Expected Output:**

```
Loaded config from ./lxfu.conf
Loading/capturing face...
⚠ Warning: --preview requested but no display detected (headless system)
⚠ Falling back to instant capture mode...
✓ Frame captured (instant mode)
Image loaded: 640x480
Extracting face embedding...
✓ Enrollment successful!
```

**Expected Behavior:**

- ✅ No window attempt
- ✅ Warning message displayed
- ✅ Falls back to instant capture
- ✅ Operation completes successfully
- ✅ No errors or crashes

---

### Test 3: SSH Without X11 Forwarding

**Setup:**

```bash
ssh user@localhost 'echo $DISPLAY'  # Should be empty
```

**Test:**

```bash
ssh user@localhost './build/bin/lxfu --preview enroll /dev/video0 test3'
```

**Expected Output:**

```
⚠ Warning: --preview requested but no display detected (headless system)
⚠ Falling back to instant capture mode...
✓ Frame captured (instant mode)
✓ Enrollment successful!
```

**Expected Behavior:**

- ✅ Detects headless environment
- ✅ Graceful fallback
- ✅ No GUI errors
- ✅ Works over SSH

---

### Test 4: SSH With X11 Forwarding

**Setup:**

```bash
ssh -X user@localhost 'echo $DISPLAY'  # Should show localhost:10.0 or similar
```

**Test:**

```bash
ssh -X user@localhost './build/bin/lxfu --preview enroll /dev/video0 test4'
```

**Expected Output:**

```
Preview mode: Press SPACE to capture, ESC to cancel...
✓ Frame captured!
✓ Enrollment successful!
```

**Expected Behavior:**

- ✅ Window opens on local display
- ✅ Preview works over network
- ✅ Capture works correctly
- ✅ Window closes cleanly

---

### Test 5: Image File with Preview (Desktop)

**Setup:**

```bash
export DISPLAY=:0
```

**Test:**

```bash
./build/bin/lxfu --preview enroll face.jpg test5
```

**Expected Output:**

```
Loaded config from ./lxfu.conf
Loading/capturing face...
Loaded image. Press any key to continue...
Image loaded: 800x600
Extracting face embedding...
✓ Enrollment successful!
```

**Expected Behavior:**

- ✅ Window shows image
- ✅ Waits for keypress
- ✅ Continues after key
- ✅ Window closes cleanly

---

### Test 6: Image File with Preview (Headless)

**Setup:**

```bash
unset DISPLAY
```

**Test:**

```bash
./build/bin/lxfu --preview enroll face.jpg test6
```

**Expected Output:**

```
Loaded config from ./lxfu.conf
Loading/capturing face...
⚠ Warning: --preview requested but no display detected (headless system)
⚠ Skipping image preview...
Image loaded: 800x600
Extracting face embedding...
✓ Enrollment successful!
```

**Expected Behavior:**

- ✅ Skips preview window
- ✅ Warning message shown
- ✅ Continues processing
- ✅ Operation successful

---

### Test 7: Headless Mode (Default Behavior)

**Setup:**

```bash
unset DISPLAY  # Not needed, but confirms headless
```

**Test:**

```bash
./build/bin/lxfu enroll /dev/video0 test7  # No --preview flag
```

**Expected Output:**

```
Loaded config from ./lxfu.conf
Loading/capturing face...
✓ Frame captured (instant mode)
Image loaded: 640x480
Extracting face embedding...
✓ Enrollment successful!
```

**Expected Behavior:**

- ✅ Instant capture
- ✅ No window attempts
- ✅ No GUI dependencies
- ✅ Fast execution

---

### Test 8: Query with Preview (Desktop)

**Setup:**

```bash
export DISPLAY=:0
```

**Test:**

```bash
./build/bin/lxfu --preview query /dev/video0
```

**Expected Output:**

```
Loaded config from ./lxfu.conf
Loading/capturing face...
Preview mode: Press SPACE to capture, ESC to cancel...
✓ Frame captured!
Image loaded: 640x480
Extracting face embedding...
Searching for similar faces...

✓ Face recognized!
  Name: test1
  Similarity: 94.23%
  Face ID: 0
```

**Expected Behavior:**

- ✅ Preview window works
- ✅ SPACE captures frame
- ✅ Recognition succeeds
- ✅ Clean window cleanup

---

### Test 9: Query with Preview (Headless)

**Setup:**

```bash
unset DISPLAY
```

**Test:**

```bash
./build/bin/lxfu --preview query /dev/video0
```

**Expected Output:**

```
⚠ Warning: --preview requested but no display detected (headless system)
⚠ Falling back to instant capture mode...
✓ Frame captured (instant mode)
Searching for similar faces...
✓ Face recognized!
```

**Expected Behavior:**

- ✅ Falls back gracefully
- ✅ Recognition works
- ✅ No errors

---

### Test 10: ESC Key Cancellation

**Setup:**

```bash
export DISPLAY=:0
```

**Test:**

```bash
./build/bin/lxfu --preview enroll /dev/video0 test10
# Press ESC when window appears
```

**Expected Output:**

```
Preview mode: Press SPACE to capture, ESC to cancel...
Error during enrollment: Capture cancelled by user
```

**Expected Behavior:**

- ✅ Window appears
- ✅ ESC key detected
- ✅ Operation cancelled
- ✅ Clean exit
- ✅ No segfault

---

### Test 11: Display Failure Mid-Operation

**Scenario:** Display server dies while preview window is open

**Test:**

```bash
# This is hard to test automatically, but code handles it
./build/bin/lxfu --preview enroll /dev/video0 test11
```

**Expected Behavior:**

- ✅ Catches cv::Exception
- ✅ Falls back to instant capture
- ✅ Warning message shown
- ✅ Operation completes

---

### Test 12: Mixed Environment Script

**Script: `test_script.sh`**

```bash
#!/bin/bash
# This script should work on both desktop and server

echo "Testing LXFU in current environment..."
./build/bin/lxfu --preview enroll /dev/video0 script_test

if [ $? -eq 0 ]; then
    echo "✓ Enrollment successful!"
else
    echo "✗ Enrollment failed!"
    exit 1
fi
```

**Test on Desktop:**

```bash
export DISPLAY=:0
bash test_script.sh
```

**Expected:** Preview window opens, works normally

**Test on Server:**

```bash
unset DISPLAY
bash test_script.sh
```

**Expected:** Falls back gracefully, completes successfully

---

## Automated Test Script

```bash
#!/bin/bash
# test_lxfu.sh - Automated test suite

set -e
trap 'echo "Test failed at line $LINENO"' ERR

echo "=== LXFU Test Suite ==="
echo

# Build first
echo "Building LXFU..."
mkdir -p build && cd build
cmake .. > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1
cd ..

# Test 1: Headless mode (default)
echo "Test 1: Headless mode (default)"
unset DISPLAY
unset WAYLAND_DISPLAY
./build/bin/lxfu enroll test_face.jpg test1 > /tmp/lxfu_test1.log 2>&1
if grep -q "✓ Enrollment successful" /tmp/lxfu_test1.log; then
    echo "✓ PASS: Headless enrollment"
else
    echo "✗ FAIL: Headless enrollment"
    cat /tmp/lxfu_test1.log
    exit 1
fi

# Test 2: Preview on headless (should fallback)
echo "Test 2: Preview on headless (should fallback)"
./build/bin/lxfu --preview enroll test_face.jpg test2 > /tmp/lxfu_test2.log 2>&1
if grep -q "⚠ Warning: --preview requested but no display detected" /tmp/lxfu_test2.log && \
   grep -q "✓ Enrollment successful" /tmp/lxfu_test2.log; then
    echo "✓ PASS: Graceful fallback"
else
    echo "✗ FAIL: Fallback not working"
    cat /tmp/lxfu_test2.log
    exit 1
fi

# Test 3: Query headless
echo "Test 3: Query headless"
./build/bin/lxfu query test_face.jpg > /tmp/lxfu_test3.log 2>&1
if grep -q "✓ Face recognized" /tmp/lxfu_test3.log; then
    echo "✓ PASS: Query successful"
else
    echo "✗ FAIL: Query failed"
    cat /tmp/lxfu_test3.log
    exit 1
fi

# Test 4: Preview query on headless (should fallback)
echo "Test 4: Preview query on headless (should fallback)"
./build/bin/lxfu --preview query test_face.jpg > /tmp/lxfu_test4.log 2>&1
if grep -q "⚠ Warning: --preview requested but no display detected" /tmp/lxfu_test4.log; then
    echo "✓ PASS: Query fallback working"
else
    echo "✗ FAIL: Query fallback not working"
    cat /tmp/lxfu_test4.log
    exit 1
fi

echo
echo "=== All tests passed! ==="
```

---

## Manual Testing Checklist

- [ ] Desktop enrollment with preview
- [ ] Desktop query with preview
- [ ] Headless enrollment (no preview flag)
- [ ] Headless query (no preview flag)
- [ ] Headless with preview flag (fallback test)
- [ ] SSH without X11 forwarding
- [ ] SSH with X11 forwarding
- [ ] ESC key cancellation
- [ ] SPACE key capture
- [ ] Image file preview (desktop)
- [ ] Image file preview (headless fallback)
- [ ] No segfaults in any scenario
- [ ] Clean window cleanup
- [ ] Warning messages clear

---

## Performance Testing

### Headless Performance

```bash
time ./build/bin/lxfu enroll /dev/video0 perf_test1
# Expected: < 1 second startup overhead
```

### Preview Performance

```bash
export DISPLAY=:0
time ./build/bin/lxfu --preview enroll /dev/video0 perf_test2
# Expected: < 100ms additional overhead for window creation
```

### Fallback Performance

```bash
unset DISPLAY
time ./build/bin/lxfu --preview enroll /dev/video0 perf_test3
# Expected: Negligible overhead for display detection
```

## Face Service Tests

### Test 7: DBus Verification (CLI client)

**Setup:**

```bash
cmake --build build --target lxfu_face_service
./build/bin/lxfu_face_service &
SERVICE_PID=$!
```

**Client (busctl example):**

```bash
busctl call dev.nabeeladzan.lxfu /dev/nabeeladzan/lxfu/Device0 dev.nabeeladzan.lxfu.Device Claim
busctl call dev.nabeeladzan.lxfu /dev/nabeeladzan/lxfu/Device0 dev.nabeeladzan.lxfu.Device VerifyStart s any
# listen for status updates in another terminal
busctl monitor dev.nabeeladzan.lxfu | rg VerificationStatus
```

**Expected Behavior:**

- ✅ `VerificationStatus` emits `verify-started`
- ✅ Matches emit `verify-match <profile>:<score>`
- ✅ Absent/failed captures emit `verify-no-face` or `verify-no-match`
- ✅ `VerifyStop` triggers `verify-cancelled`
- ✅ `Release` frees the device cleanly

Shutdown the daemon after testing:

```bash
kill $SERVICE_PID
```
