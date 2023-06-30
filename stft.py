import numpy as np
import scipy.io.wavfile
import scipy.signal
import librosa
import soundfile as sf

def process_audio(input_file, output_file):
    # Load audio file and convert to mono
    y, sr = librosa.load(input_file, sr=None, mono=True)

    # Resample to 44.1kHz if necessary
    if sr != 44100:
        y = librosa.resample(y, sr, 44100)
        sr = 44100

    # Define STFT parameters
    window_length = int(sr * .01)  # 10ms window
    window_hop = window_length // 2  # 50% overlap

    # Compute STFT
    f, t, stft = scipy.signal.stft(y, sr, nperseg=window_length, noverlap=window_hop)

    # Compute amplitude and phase
    amplitude = np.abs(stft)
    phase = np.angle(stft)

    # Compute their derivatives
    amplitude_derivative = np.diff(amplitude, axis=-1) / (t[1] - t[0])
    phase_derivative = np.diff(np.unwrap(phase), axis=-1) / (t[1] - t[0])

    # Reconstruct amplitude and phase from their derivatives
    amplitude_reconstructed = np.cumsum(np.pad(amplitude_derivative, ((0, 0), (1, 0))), axis=-1)
    phase_reconstructed = np.cumsum(np.pad(phase_derivative, ((0, 0), (1, 0))), axis=-1)

    # Construct the STFT from the reconstructed amplitude and phase
    stft_reconstructed = amplitude_reconstructed * np.exp(1j * phase_reconstructed)

    # Compute the inverse STFT
    t, y_reconstructed = scipy.signal.istft(stft_reconstructed, sr, nperseg=window_length, noverlap=window_hop)

    # Write to output file
    sf.write(output_file, y_reconstructed, sr)

# Test the function
process_audio('input.wav', 'output.wav')
