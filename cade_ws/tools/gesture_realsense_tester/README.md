# Gesture RealSense Tester

Standalone tester for the CADE `gesture.py` rule module.

## Run

```bash
cd ~/Desktop/gesture_realsense_tester
conda run -n task3 python gesture_realsense_test.py
```

Useful options:

```bash
conda run -n task3 python gesture_realsense_test.py --flip
conda run -n task3 python gesture_realsense_test.py --t-forearm 300 --t-wrist 500 --jump-threshold 50
conda run -n task3 python gesture_realsense_test.py --serial REALSENSE_SERIAL
```

Keys:

- `q` or `Esc`: quit
- `r`: clear temporal ring buffer
- `s`: save screenshot under `~/Desktop/gesture_realsense_tester/screenshots`

The script uses RealSense color frames only. It runs MediaPipe Pose and Hands on
the live image, then applies the vendored static and temporal gesture rules.
