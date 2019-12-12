import pickle
import numpy
from music21 import instrument, note, stream, chord
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Activation

# use GPU device 0
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow.keras
config = tf.ConfigProto( device_count = {'GPU': 0})
sess = tf.Session(config = config)
tensorflow.keras.backend.set_session(sess)

def generate():
    """
    Main function to generate a midi file.
    """
    #load the notes used to train the model
    with open('data/schubert', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Number of unique notes
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    #print("made sequences")
    model = create_network(normalized_input, n_vocab)
    #print("made model")
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    #print("generated notes")
    create_midi(prediction_output)
    #print("made midi")

def prepare_sequences(notes, pitchnames, n_vocab):
    """
    Process list of notes into matrix of ints for input into model.  
    """
    # map notes to unique ints
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # cut notes into 100 length chuncks
    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input so it aligns w the model
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab):
    """
    Make the model - can be GRU or LSTM
    """
    model = Sequential()
    #model.add(LSTM(
    #    512,
    #    input_shape=(network_input.shape[1], network_input.shape[2]),
    #    recurrent_dropout=0.3,
    #    return_sequences=True
    #))
    #model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))

    model.add(GRU(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(GRU(512, return_sequences=False, recurrent_dropout=0.3,))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load weights for the model
    model.load_weights('weights-improvement-GRU-123-1.8256.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """
    Generate notes with the model and save music as list of ints.
    """
    # pick a note(intdex) from the input as starting note for generation
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        if note_index % 10 == 0:
            print("on index: ", note_index)
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """
    Take list of ints from output of model, convert to notes and save to mid file.
    """
    offset = 0
    output_notes = []

    # make a note / chord based on ints from output
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            # parse chord
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # parse note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increment offset so notes don't stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    # save midi to a file
    midi_stream.write('midi', fp='GRU_test_output.mid')

if __name__ == '__main__':
    generate()
