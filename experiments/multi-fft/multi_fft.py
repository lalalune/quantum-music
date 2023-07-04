import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment

def stft_method(y, sr):
    # Compute STFT
    stft = librosa.stft(y)

    # Invert the STFT
    y_reconstructed = librosa.istft(stft)

    return y_reconstructed

def dft_method(y, sr):
    # Compute DFT
    dft = np.fft.fft(y)

    # Invert the DFT
    y_reconstructed = np.fft.ifft(dft)

    return y_reconstructed

def fft_method(y, sr):
    # Compute FFT
    fft = np.fft.fft(y)

    # Invert the FFT
    y_reconstructed = np.fft.ifft(fft)

    return y_reconstructed

def generate_spectrogram(y, sr, output_file):
    # Generate spectrogram
    plt.figure(figsize=(10, 10))
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap='inferno', sides='default', mode='default', scale='dB');
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0,0)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    plt.close()

def process_audio(input_file, output_file, mix=100):
    # Load audio file and convert to mono
    y, sr = librosa.load(input_file, sr=None, mono=True)
    original_y = np.copy(y)

    # Resample to 44.1kHz if necessary
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100

    for method, name in zip([stft_method, dft_method, fft_method], ["stft", "dft", "fft"]):
        # Compute and invert transformation
        y_reconstructed = method(y, sr)

        # Pad the shorter array with zeros up to the length of the longer array
        len_diff = len(original_y) - len(y_reconstructed)
        if len_diff > 0:
            y_reconstructed = np.pad(y_reconstructed, (0, len_diff))
        elif len_diff < 0:
            original_y = np.pad(original_y, (0, -len_diff))

        # Mix original and transformed audio
        mix_b = mix / 100.0
        mix_a = 1.0 - mix_b
        mixed_y = original_y * mix_a + np.real(y_reconstructed) * mix_b

        # Normalize audio signals
        mixed_y = librosa.util.normalize(mixed_y)

        # Write to output file
        sf.write(f"{name}_{output_file}", mixed_y, sr)

        # Generate spectrogram images
        generate_spectrogram(mixed_y, sr, f"{name}_{output_file}.png")

# Test the function
process_audio('../../wavs/input.wav', 'output.wav', mix=100)
