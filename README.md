# Speech Emotion Recognition

A machine learning project that predicts the emotion behind a short speech clip — happy, sad, angry, calm, and more — using audio feature extraction and a neural network classifier.

Built as a learning project to understand the fundamentals of audio ML: turning raw sound into structured features, training a classifier, and evaluating (and stress-testing) how well it actually generalizes.

## How it works

1. **Feature extraction** — raw audio is converted into a fixed-length numeric feature vector using three classic audio features (via [`librosa`](https://librosa.org/)):
   - **MFCCs** (Mel-Frequency Cepstral Coefficients) — capture the timbre/tone-color of a voice
   - **Chroma** — captures pitch-class content
   - **Mel spectrogram** — a frequency-vs-time energy map scaled to human hearing

   Each is averaged over time, producing one **180-number vector** per audio clip regardless of its original length.

2. **Scaling** — features are standardized (`StandardScaler`) so no single feature dominates training due to differences in raw scale.

3. **Classification** — an `MLPClassifier` (a small feedforward neural network, scikit-learn) is trained on the labeled feature vectors to predict one of 8 emotions.

## Dataset

[RAVDESS](https://zenodo.org/records/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song) — speech-only subset. 1,440 audio clips from 24 actors, each labeled with one of 8 emotions via a structured filename code (e.g. `03-01-06-01-02-01-12.wav`).

**Note:** the dataset is not included in this repo (it's ~215MB). Download `Audio_Speech_Actors_01-24.zip` from the link above and extract it into a `data/` folder at the project root, so you end up with `data/Actor_01/`, `data/Actor_02/`, ... `data/Actor_24/`.

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd speech-emotion-recognition

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Download and extract the RAVDESS dataset into `data/` as described above.

## Usage

```bash
python main.py
```

This runs the full pipeline: loads the dataset, extracts features from all 1,440 clips, splits into train/test sets, trains the classifier, and prints accuracy and a per-emotion breakdown.

To test on a new audio file, update the `new_file` variable in `main.py` to point to your own `.wav` file.

## Results

- **Accuracy: ~68.6%** on held-out test data (360 clips), against a random-guess baseline of ~12.5% across 8 classes.
- Strongest performance on **calm**, **angry**, and **surprised** — emotions with distinctive acoustic signatures.
- Weakest on **fearful** (51% recall), frequently confused with sad, happy, and disgust.
- **Calm and neutral are commonly confused with each other** — an expected result, since both are low-energy, flat-delivery emotions that are acoustically similar even to human listeners.

## Limitations

- **Trained only on RAVDESS's 24 actors** reading a fixed, scripted phrase in a controlled studio setting. It does not generalize well to spontaneous, real-world speech from unseen speakers — testing on my own recorded voice, the model misclassified an intended "angry" clip as "calm."
- **Sensitive to clip length.** Training clips are ~3–5 seconds; features are time-averaged, so significantly longer clips (with pauses, silence, or varying delivery) dilute the signal the model learned on.
- **Neutral is underrepresented** in RAVDESS (no high-intensity variant exists for it), which likely affects the model's confidence on that class.

## Tech stack

Python · librosa · scikit-learn · numpy

## Acknowledgments

Built following the general approach outlined in [DataFlair's Speech Emotion Recognition tutorial](https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/), implemented from scratch for learning purposes.
