import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image

def audio_to_spectrogram(y, sr, output_file):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    plt.close()

def spectrogram_to_audio(input_file, output_file):
    image = Image.open(input_file).convert('L')
    data = np.array(image).astype(np.float32)

    # Convert the 8-bit data to floating-point data
    data = (data / 255.0) ** 2

    # Invert the spectrogram
    data_inverted = np.exp(data)

    # Perform the Griffin-Lim algorithm to recover audio
    p = np.random.uniform(-np.pi, np.pi, size=data_inverted.shape)

    for i in range(500):
        S_complex = data_inverted * np.exp(1j * p)
        y = librosa.istft(S_complex)

        p = np.angle(librosa.stft(y))

    sf.write(output_file, y, 44100)

def process_audio(input_file):
    # Load audio file and convert to mono
    y, sr = librosa.load(input_file, sr=None, mono=True)

    # Resample to 44.1kHz if necessary
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100

    spectrogram_file = "spectrogram.png"

    # Generate spectrogram image if it doesn't exist
    if not os.path.isfile(spectrogram_file):
        audio_to_spectrogram(y, sr, spectrogram_file)

    # Convert spectrogram back to audio
    spectrogram_image = Image.open(spectrogram_file).convert('L')
    data_inverted = np.array(spectrogram_image).astype(np.float32)
    p = np.random.uniform(-np.pi, np.pi, size=data_inverted.shape)

    for i in range(500):
        S_complex = data_inverted * np.exp(1j * p)
        y = librosa.istft(S_complex)

        p = np.angle(librosa.stft(y))

    sf.write("output.wav", y, 44100)

# Test the function
process_audio('../../wavs/input.wav')
