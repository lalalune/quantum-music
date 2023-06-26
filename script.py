# Import the necessary libraries
import cirq
import numpy as np
import wave
import os
import soundfile as sf
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Define a function to read a wave file into a numpy array
def read_wav_file(filename):
    with wave.open(filename, 'rb') as wave_file:
        params = wave_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]

        str_data = wave_file.readframes(nframes)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))

    return wave_data, framerate

# Define a function to create a quantum circuit that encodes a chunk of Fourier data
def create_circuit(fft_data_chunk, qubits):
    print('Creating circuit')
    circuit = cirq.Circuit()

    # Apply QFT
    for i, _ in enumerate(qubits):
        for j in range(i):
            circuit.append(cirq.CZPowGate(exponent=-1/2**(i - j))(qubits[i], qubits[j]))
        circuit.append(cirq.H(qubits[i]))
    for i in range(len(qubits) // 2):
        circuit.append(cirq.SWAP(qubits[i], qubits[-i-1]))

    # Encode the Fourier transform data
    for i, fft_coefficient in enumerate(fft_data_chunk):
        magnitude = np.abs(fft_coefficient)
        phase = np.angle(fft_coefficient)
        circuit.append(cirq.rz(magnitude)(qubits[i]))
        circuit.append(cirq.rx(phase)(qubits[i]))

    return circuit

# Read the wave data from the file
wave_data, framerate = read_wav_file('kick.wav')

# Define the number of qubits in each circuit
n_qubits = 16

# Set the max samples
MAX_SAMPLES = 1000

# Take only the first MAX_SAMPLES from the wave data
wave_data = wave_data[:MAX_SAMPLES]

# Create a list of qubits for the quantum circuit
qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]

# Determine the number of chunks
n_chunks = len(wave_data) // n_qubits

# Make sure /data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Create a .wav file to hold the output
outfile = sf.SoundFile('output.wav', 'w', samplerate=framerate, channels=1, subtype='PCM_16')

# Initialize an empty list to hold the output data for spectrogram
output_data = []

# Iterate over each chunk
for i in range(n_chunks):
    # Get the wave data chunk
    wave_data_chunk = wave_data[i * n_qubits: (i + 1) * n_qubits]

    # Perform Fourier Transform on the wave data chunk
    fft_data_chunk = fft(wave_data_chunk)

    # Create a quantum circuit for the FFT data chunk
    circuit = create_circuit(fft_data_chunk, qubits)

    # Save the circuit as a .json file
    cirq.to_json(circuit, f'data/circuit_{i}.json')

    # Simulate the quantum circuit
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    simulated_final_state = result.final_state_vector

    # Perform Inverse Fourier Transform on the simulated final state to get the simulated wave data chunk
    simulated_wave_data_chunk = ifft(simulated_final_state)

    # Normalize the simulated wave data chunk to the range [-1, 1]
    simulated_wave_data_chunk = simulated_wave_data_chunk * 1.0 / (max(abs(simulated_wave_data_chunk)))

    # Convert the simulated wave data chunk to 16-bit PCM
    simulated_wave_data_chunk_pcm = np.int16(simulated_wave_data_chunk * 32767)

    # Append the simulated wave data chunk to the output data
    output_data.extend(simulated_wave_data_chunk_pcm)

    # Write the simulated wave data chunk to the .wav file
    outfile.write(simulated_wave_data_chunk_pcm)

# Close the .wav file
outfile.close()

# Convert output data to numpy array
output_data = np.array(output_data)

# Create spectrograms
plt.specgram(wave_data, Fs=framerate)
plt.title('Input Spectrogram')
plt.savefig('input_spectrogram.png')

plt.specgram(output_data, Fs=framerate)
plt.title('Output Spectrogram')
plt.savefig('output_spectrogram.png')

print('Finished')
