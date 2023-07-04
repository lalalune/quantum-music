import cirq
import numpy as np
import pretty_midi

# Create a PrettyMIDI object
midi = pretty_midi.PrettyMIDI()

# Create an Instrument instance for a drum instrument (use 0 for 'Acoustic Grand Piano')
drum = pretty_midi.Instrument(program=0)

def generate_midi(matrix):
    # convert matrix into string
    pattern = ''
    for row in matrix:
        for column in row:
            pattern += str(column) + ' '  # append a space after each number
        pattern += '\n'

    # Split the pattern into lines
    lines = pattern.strip().split("\n")

    # Iterate over each line
    for note_number, line in enumerate(lines, start=60-24):  # 60 is middle C
        # Split the line into beats
        beats = line.split()

        # Iterate over each beat
        for beat_number, beat in enumerate(beats):
            # If the beat is a 1, add a note
            if beat == '1':
                # Create a Note instance for this note
                note = pretty_midi.Note(
                    velocity=100,  # how loud the note is
                    pitch=note_number,  # which note to play
                    start=beat_number / 4,  # when the note starts, each beat is 1 second
                    end=(beat_number + 1) / 4  # when the note ends, each beat lasts 1 second
                )

                # Add it to our drum instrument
                drum.notes.append(note)

    # Add the drum instrument to the PrettyMIDI object
    midi.instruments.append(drum)

    # Write out the MIDI data
    midi.write('output.mid')

def generate_quantum_circuit(repetitions):
    # Create a quantum circuit
    circuit = cirq.Circuit()

    # Create qubits
    qubits = [cirq.GridQubit(x, y) for x in range(4) for y in range(4)]

    # Add a Hadamard gate to each qubit
    for qubit in qubits:
        circuit.append(cirq.H(qubit))

    # Measure each qubit
    for qubit in qubits:
        circuit.append(cirq.measure(qubit))

    # Run the circuit multiple times
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=repetitions)

    # Convert the result to a numpy array
    matrix = np.empty((repetitions, len(qubits)))
    for i, measurement in enumerate(result.measurements.values()):
        for j, bit in enumerate(measurement):
            matrix[i, j] = bit

    # split into two matrices of 8x16
    matrix = np.split(matrix, 2)

    # create a third matrix where the 1s are only if both matrices have 1s in that position
    matrix1 = np.multiply(matrix[0], matrix[1])

    # create a fourth matrix where the 1s are only if both matrices have 0s in that position
    matrix2 = np.multiply(np.subtract(1, matrix[0]), np.subtract(1, matrix[1]))
    

    # combine matrix1 and matrix2 into a 8x32 matrix
    matrix = np.concatenate((matrix1, matrix2), axis=1)

    # convert values in matrix to ints
    matrix = matrix.astype(int)
    generate_midi(matrix)
    # Return the matrix
    return matrix


# Main function

def main():
    repetitions = 16
    print ('Generating a quantum circuit with ' + str(repetitions) + ' repetitions...')
    print ('Beat pattern is:')
    matrix = generate_quantum_circuit(repetitions)
    print(matrix)
    # Save the matrix to a MIDI file
    # write the matrix to a file
    with open('matrix.txt', 'w') as f:
        for row in matrix:
            for column in row:
                f.write(str(column) + ' ')
            f.write('\n')


# Call the main function if the script is run directly
if __name__ == '__main__':
    main()