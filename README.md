# gender_cv

A computer vision project for gender/person detection and tracking using YOLO models, OpenCV, and YouTube stream capture.

## Project Overview

This repository includes scripts for:
- real-time inference from a video stream (`main_inference.py`)
- webcam-based gender classification (`classify.py`)
- collecting person crops from a YouTube stream (`collect_data.py`)
- camera-based detection and tracking with SQLite logging (`capture.py`)
- initializing the analytics SQLite database (`init_db.py`)

The project uses YOLO models stored in the repository:
- `yolo11n.pt` — detector model
- `gender_yolo11.pt` / `gender_yolo11n_v2.pt` — classification/tracking models

## Requirements

The project dependencies are listed in `requirements.txt`.
Install them into a Python environment before running any script.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Initialize the SQLite database.

```powershell
python init_db.py
```

## Usage

### 1. Run stream inference

`main_inference.py` reads a YouTube stream URL, detects people, tracks them, classifies each person, and writes a record to `cv_analytics.db`.

```powershell
python main_inference.py
```

- Update `YOUTUBE_URL` in `main_inference.py` to change the input stream.
- Change `ROI`, `DRAW_ROI`, and `DRAW_BOXES` for bounding-box behavior.
- The script saves detected track IDs and classification labels to `people_tracks`.

### 2. Run webcam classification

`classify.py` opens the local camera and classifies each captured frame using `gender_yolo11n_v2.pt`.

```powershell
python classify.py
```

Press `q` to exit.

### 3. Collect training data from a stream

`collect_data.py` captures person crops from a YouTube stream and saves them under `dataset_raw`.

```powershell
python collect_data.py
```

- Update `YOUTUBE_URL` and `ROI` to tune what is collected.
- Crops are saved automatically with a unique ID file name.

### 4. Use camera tracking/demo mode

`capture.py` performs detection and tracking from a local camera and logs results to `cv_analytics.db`.

```powershell
python capture.py
```

## Database

The repository stores results in `cv_analytics.db`. The schema for `people_tracks` is:
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `track_id` INTEGER NOT NULL
- `label` TEXT NOT NULL
- `timestamp` DATETIME DEFAULT current local time

## Notes

- `cookies.txt` and `cookie.json` are referenced by the YouTube downloader configuration.
- `main.py` is a default sample script and is not used by the main CV workflow.
- Ensure the required YOLO model files are present in the repository root.

## License

No license is specified in this repository.
# ai-gender-detector
