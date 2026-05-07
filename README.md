# Sign-Bridge
Real-Time Hybrid ASL Recognition System

Sign-Bridge is a real-time American Sign Language recognition system for webcam-based recognition of both static fingerspelled letters and dynamic word gestures. The project uses a hybrid pipeline built around MediaPipe landmark extraction, a Random Forest letter model, dynamic word modeling, and layered correction using fuzzy matching, Local Gemma, and optional OpenAI fallback.

## Overview

The system is designed to recognize:
- Static ASL fingerspelling letters
- Dynamic ASL words and short gesture sequences
- Sentence-style output using trained space handling
- Corrected transcript output with timing and evaluation logs
- Real-time confidence, explanation, and debug overlays

At a high level, the demo:
1. Captures webcam frames with OpenCV
2. Extracts hand and face landmarks with MediaPipe Tasks
3. Routes input into letter mode or word mode
4. Applies the appropriate classifier pipeline
5. Builds transcript output and correction candidates
6. Displays live overlays and writes evaluation logs

## Key Features

### Real-Time Recognition
- Live webcam recognition using OpenCV
- Real-time demo path for common hardware
- Confidence thresholds, smoothing, and stabilizing logic
- Motion-aware routing for word handling
- Presentation-friendly overlays, corrected transcript display, and timing logs

### Letter Recognition
- Uses normalized hand landmarks only
- Flattens 21 hand landmarks into a 63-feature vector
- Random Forest letter classifier trained from collected samples
- Majority-vote smoothing and release-based stabilization
- Supports both a true space label and an equals-style sentence separator in the transcript logic

### Word Recognition
- Uses temporal handling for dynamic signs
- Includes a hand-sequence word path
- Includes a richer sequence-model path
- Supports hand-plus-face processing in the broader word pipeline

### Hybrid Correction
Fingerspelled output can be refined in stages:
1. Local fuzzy matching
2. Local Gemma correction
3. Optional OpenAI fallback when needed

### Visual Output and Evaluation
- Transcript display
- Confidence display
- Explanation text
- Motion and debug overlay support
- Performance logging for scenario type, backend timings, and final outputs

## System Architecture

```text
Webcam frame / still input
        |
        v
MediaPipe Tasks hand + face landmark extraction
        |
        v
Feature preparation
- letter: normalized hand landmarks to 63D vector
- word: temporal sequence features
        |
        v
Routing logic
- letter mode
- word mode
- motion-aware word branching
        |
        v
Model inference
- letter: Random Forest
- word: dynamic word model
        |
        v
Post-processing
- thresholds
- smoothing
- transcript assembly
- fuzzy correction
- Local Gemma
- optional OpenAI fallback
        |
        v
OpenCV overlay + logs
```

## Requirements

Python 3.11 or newer is recommended.

Install core dependencies:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib torch python-dotenv
```

Optional API support:

```bash
pip install openai
```

## Setup

Create the main project folders if they do not already exist:

```text
data/
models/
logs/
```

Place the MediaPipe task files under `models/`:

```text
models/hand_landmarker.task
models/face_landmarker.task
```

If you want OpenAI fallback, place your API key in `.env`.

## Data Collection and Training

### 1. Collect letter data

```bash
python collect_asl_data.py
```

Primary output:
- `data/asl_landmarks.csv`

This stores normalized hand landmark rows for letter training. The letter model is built from 63 hand features per sample.

### 2. Train the letter model

```bash
python train_asl_model.py
```

Primary outputs:
- `models/asl_model.pkl`
- `models/asl_label_encoder.pkl`

### 3. Collect dynamic word data

```bash
python collect_asl_words_handseq.py
```

Optional broader word-data path:

```bash
python collect_asl_words.py
```

### 4. Train dynamic word models

```bash
python train_asl_word_handseq_model.py
```

Optional broader word-model path:

```bash
python train_asl_word_model.py
```

## Running the Demo

```bash
python Sign-Bridge-Demo.py
```

The demo currently points to:
- `models/asl_model.pkl`
- `models/asl_label_encoder.pkl`
- `models/asl_word_model.pkl`
- `models/asl_word_label_encoder.pkl`
- `models/asl_word_handseq_model.pkl`
- `models/asl_word_handseq_labels.json`
- `models/hand_landmarker.task`
- `models/face_landmarker.task`

## Correction Pipeline

The correction stack is layered:
- fuzzy matching for quick local cleanup
- Local Gemma for stronger local correction
- optional OpenAI fallback when local correction is weak or unavailable


## Demo Controls

Main demo controls used during the presentation:
- `c` clears the transcript
- `p` toggles the performance panel
- `` ` `` toggles word mode
- `r` records the custom toggle gesture template
- `F` can be used in some environments to fullscreen the OpenCV window

The demo also supports a trained actual space label ` ` and maps `=` to a word break in the transcript pipeline.

## Presentation Checklist

For the live presentation, verify these before starting:
- `models/hand_landmarker.task` and `models/face_landmarker.task` are present
- trained model files exist under `models/`
- webcam opens correctly in OpenCV
- `logs/` exists so evaluation CSV output can be written
- `.env` is populated only if OpenAI fallback is needed
- at least one test run has been done for both letter mode and word mode

## Project Structure

This is a high-level structure focused on the files and folders that matter most during setup, training, and demo use.

```text
Sign-Bridge/
|-- Sign-Bridge-Demo.py
|-- README.md
|-- .env
|
|-- collect_asl_data.py
|-- train_asl_model.py
|
|-- collect_asl_words_handseq.py
|-- train_asl_word_handseq_model.py
|
|-- collect_asl_words.py
|-- train_asl_word_model.py
|
|-- data/
|   |-- asl_landmarks.csv
|   |-- asl_words_handseq.npz
|   |-- asl_words_handseq_labels.json
|   `-- words_seq/
|
|-- models/
|   |-- hand_landmarker.task
|   |-- face_landmarker.task
|   |-- asl_model.pkl
|   |-- asl_label_encoder.pkl
|   |-- asl_word_handseq_model.pkl
|   |-- asl_word_seq_model.pt
|   `-- asl_word_seq_model_meta.json
|
`-- logs/
    `-- sign_bridge_performance.csv
```

## Typical Commands

```bash
python collect_asl_data.py
python train_asl_model.py
python collect_asl_words_handseq.py
python train_asl_word_handseq_model.py
python collect_asl_words.py
python train_asl_word_model.py
python Sign-Bridge-Demo.py
```

## Evaluation Workflow

The project now includes `evaluate_sign_bridge.py` so you can measure correction quality and end-to-end transcript quality using ground-truth CSV files instead of timing alone.

Create starter benchmark templates:

```bash
python evaluate_sign_bridge.py make-templates
```

Run the current autocorrect pipeline on a correction benchmark with `raw_input` and `expected_output` columns:

```bash
python evaluate_sign_bridge.py eval-corrections --benchmark data/correction_benchmark_template.csv
```

Score an existing `logs/sign_bridge_performance.csv` file against a ground-truth correction benchmark:

```bash
python evaluate_sign_bridge.py score-logs --log-csv logs/sign_bridge_performance.csv --ground-truth-csv data/correction_benchmark_template.csv
```

Measure end-to-end transcript quality using exact match, character error rate, and word error rate:

```bash
python evaluate_sign_bridge.py eval-transcripts --benchmark data/transcript_benchmark_template.csv
```

## Notes

- Letter recognition uses hand landmarks only.
- The broader word path uses temporal data and can incorporate hand and face information.
- Performance results are logged under `logs/` for evaluation and scenario comparison.

## Author

Nomar Rodriguez  
M.S. Computer Science  
Old Dominion University
