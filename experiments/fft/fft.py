import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment

def process_audio(input_file, output_file, mix=100):
    # Load audio file and convert to mono
    y, sr = librosa.load(input_file, sr=None, mono=True)
    original_y = np.copy(y)

    # Resample to 44.1kHz if necessary
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100

    # Compute FFT
    fft = np.fft.fft(y)

    # Compute amplitude and phase
    amplitude = np.abs(fft)
    phase = np.angle(fft)

    # Compute their derivatives
    amplitude_derivative = np.diff(amplitude) / 1
    phase_derivative = np.diff(np.unwrap(phase)) / 1

    # Reconstruct amplitude and phase from their derivatives
    amplitude_reconstructed = np.cumsum(np.pad(amplitude_derivative, (1, 0)))
    phase_reconstructed = np.cumsum(np.pad(phase_derivative, (1, 0)))

    # Construct the FFT from the reconstructed amplitude and phase
    fft_reconstructed = amplitude_reconstructed * np.exp(1j * phase_reconstructed)

    # Compute the inverse FFT
    y_reconstructed = np.fft.ifft(fft_reconstructed)

    # Pad the shorter array with zeros up to the length of the longer array
    len_diff = len(original_y) - len(y_reconstructed)
    if len_diff > 0:
        y_reconstructed = np.pad(y_reconstructed, (0, len_diff))
    elif len_diff < 0:
        original_y = np.pad(original_y, (0, -len_diff))

    # Mix original and FFT-processed audio
    mix_b = mix / 100.0
    mix_a = 1.0 - mix_b
    mixed_y = original_y * mix_a + np.real(y_reconstructed) * mix_b

    # Normalize audio signals
    mixed_y = librosa.util.normalize(mixed_y)

    # Write to output file
    sf.write(output_file, mixed_y, sr)

    # Generate spectrogram images
    orig_spectrogram = np.abs(librosa.stft(original_y))

    plt.figure(figsize=(10, 10))
    plt.specgram(original_y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap='inferno', sides='default', mode='default', scale='dB');
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0,0)
    plt.savefig("fft.png", bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    plt.close()

# Test the function
process_audio('wavs/input.wav', 'fft.wav', mix=100)
