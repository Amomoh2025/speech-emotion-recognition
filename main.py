import librosa
import numpy as np
import glob
import os

def extract_features(file_path):
    # Load audio, resampled to a consistent rate (librosa's default: 22050 Hz)
    audio, sample_rate = librosa.load(file_path)

    # MFCCs: timbre / tone-color, 40 coefficients per time frame
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # collapse time axis -> fixed-length vector

    # Chroma: pitch-class content, needs a spectrogram first
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Mel spectrogram: perceptual frequency-energy map
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)

    # Concatenate into a single fixed-length feature vector
    return np.hstack([mfccs_mean, chroma_mean, mel_mean])


# Quick test on our one sample file
file_path = "data/Actor_01/03-01-01-01-01-01-01.wav"
features = extract_features(file_path)
print("Feature vector shape:", features.shape)
print("First 10 values:", features[:10])

# Map the emotion code (from filename) to a readable label
emotion_labels = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_data():
    X = []  # feature vectors
    y = []  # emotion labels

    for file_path in glob.glob("data/Actor_*/*.wav"):
        # filename looks like: 03-01-06-01-02-01-12.wav
        filename = os.path.basename(file_path)
        parts = filename.split("-")
        emotion_code = parts[2]  # 3rd number = emotion
        emotion = emotion_labels[emotion_code]

        features = extract_features(file_path)

        X.append(features)
        y.append(emotion)

    return np.array(X), np.array(y)


X, y = load_data()
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique emotions found:", set(y))

#train/Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learn mean/std from training data, then scale it
X_test = scaler.transform(X_test)        # apply that same scaling to test data (no re-fitting)

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(300,),  # one hidden layer, 300 neurons
    max_iter=500,               # max training iterations
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete.")

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=list(emotion_labels.values())))

#final test on folder
new_file = "data/Actor_13/03-01-05-02-01-01-13.wav"

# Extract features the exact same way as training data
new_features = extract_features(new_file)

# Reshape: extract_features returns a 1D array (180,),
# but the model expects a 2D array — a batch of samples, even if it's just one
new_features = new_features.reshape(1, -1)

# Scale using the ALREADY-FITTED scaler (transform, not fit_transform)
new_features_scaled = scaler.transform(new_features)

# Predict
prediction = model.predict(new_features_scaled)
print("Predicted emotion:", prediction[0])