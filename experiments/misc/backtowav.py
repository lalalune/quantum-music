# Import the necessary libraries
from cirq.contrib.qasm_import import circuit_from_qasm
import cirq
import numpy as np
import glob
import wave
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# Get a list of all .qasm files in the /data directory
filenames = glob.glob('data/*.qasm')

# sort filenames alpha-numerically, so that _0, _1, _2, etc. are in order, NOT _0, _10, _11, etc.
filenames.sort(key=natural_keys)

# Open the output .wav file
with wave.open('output.wav', 'wb') as wave_file:
    # Set the parameters
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(44100)

    # Iterate over each .qasm file
    for filename in filenames:
        print(filename)
        # Read the .qasm file
        with open(filename, 'r') as f:
            qasm = f.read()

        # Convert the .qasm file to a Cirq circuit
        circuit = circuit_from_qasm(qasm)

        print('circuit')
        print(circuit)

        # Simulate the circuit and get the final state vector
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        final_state_vector = result.final_state_vector

        # Convert the final state vector to wave data by taking the amplitude of each state
        wave_data_chunk = np.abs(final_state_vector)

        # Normalize the wave data to the range [-1, 1]
        wave_data_chunk = wave_data_chunk * 1.0 / (max(abs(wave_data_chunk)))

        # Convert the wave data to 16-bit PCM
        wave_data_pcm = np.int16(wave_data_chunk * 32767)

        # Write the wave data chunk to the .wav file
        wave_file.writeframes(wave_data_pcm.tobytes())

print('Finished')
