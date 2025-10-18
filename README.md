# Track Layout Maker — Streamlit (Metric) v0.1

Run locally in your browser with Streamlit.

## Install
```bash
python3 -m pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

- Use the sidebar to pick shape, dimensions, stock lengths, joins/ends.
- Add mid-run components as `position_m:PARTNO` lines.
- Click **Generate PDF** to download a single-page PDF.
- For batch, upload a CSV like `layouts_sample.csv` and click **Generate batch PDF**.

### CSV columns
```
LayoutName,Shape,Length,Width,Depth,Stock,MaxRun,StartEnd,EndEnd,Corner1,Corner2,Corner3,MidComponents
Straight-5.84,Straight,5.84,,,,"2,1",,End cap,End cap,,,,1.50:FEED-TEE|3.25:SENSOR-IR
L-3x2,L,3.0,2.0,,"2,1",,End cap,End cap,Plain join,,
```

### Notes
- Click-to-place mid components is not in v0.1. Manual position entry is supported. We can add a custom component later to capture mouse clicks.
- T/X shapes can be approximated by multiple Straights for now.
