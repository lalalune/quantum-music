from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import write
import argparse

def convert_to_stereo(audio):
    if audio.channels == 2:
        return audio
    return audio.set_channels(2)

def stereo_to_numpy(audio):
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        return np.reshape(samples, (-1, 2))
    else:
        return np.repeat(samples[:, np.newaxis], 2, axis=1)

def normalize_audio(data):
    gain = 32767.0 / np.max(np.abs(data))
    return data * gain

def mix_audio_files(input_path_a="input_a.wav", input_path_b="input_b.wav", output_path="mixed.wav", mix_percentage=50):
    # Read the audio files
    audio1 = AudioSegment.from_wav(input_path_a)
    audio2 = AudioSegment.from_wav(input_path_b)

    # Convert to stereo if they are not already
    audio1_stereo = convert_to_stereo(audio1)
    audio2_stereo = convert_to_stereo(audio2)

    # Make sure they have the same frame rate
    if audio1_stereo.frame_rate != audio2_stereo.frame_rate:
        frame_rate = max(audio1_stereo.frame_rate, audio2_stereo.frame_rate)
        audio1_stereo = audio1_stereo.set_frame_rate(frame_rate)
        audio2_stereo = audio2_stereo.set_frame_rate(frame_rate)

    data1 = stereo_to_numpy(audio1_stereo)
    data2 = stereo_to_numpy(audio2_stereo)

    # Normalize audio data
    data1 = normalize_audio(data1)
    data2 = normalize_audio(data2)

    # Make sure they have the same length
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]

    # Mix the audios
    mix_b = mix_percentage / 100.0
    mix_a = 1.0 - mix_b
    mixed = np.array(data1 * mix_a + data2 * mix_b, dtype=np.int16)

    # Save the result
    write(output_path, frame_rate, mixed)

# Command-line arguments parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mix two audio files.")
    parser.add_argument('--inputa', type=str, default='input_a.wav', help='Path to input file a')
    parser.add_argument('--inputb', type=str, default='input_b.wav', help='Path to input file b')
    parser.add_argument('--output', type=str, default='mixed.wav', help='Path to output file')
    parser.add_argument('--mix_percentage', type=int, default=50, help='Mix percentage for file b (0-100)')
    args = parser.parse_args()

    mix_audio_files(args.inputa, args.inputb, args.output, args.mix_percentage)
