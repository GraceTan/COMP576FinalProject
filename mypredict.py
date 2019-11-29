""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Activation

from mymodel import seq2seq, EncoderRNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    print("made sequences")
    # model = create_network(normalized_input, n_vocab)
    encoder = EncoderRNN(358, 512)
    decoder = DecoderRNN(512, 358)
    model = seq2seq(encoder, decoder, device).to(device)
    state = torch.load('modelstate.pth')
    model.load_state_dict(state['state_dict'])
    print("made model")
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    print("generated notes")
    create_midi(prediction_output)
    print("made midi")

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def undo_categorical(y, dtype='float32'):
    return np.argmax(y, axis=1)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length - 1, 1):
        sequence_in = notes[i:i + sequence_length]  # 0:100
        sequence_out = notes[i+1: i + sequence_length + 1]  # 1:101
        network_input.append(to_categorical([note_to_int[char] for char in sequence_in], 358))
        network_output.append(to_categorical([note_to_int[char] for char in sequence_out], 358))

    for i in range(len(network_input)):
        if len(network_input[i]) != 100:
            print("SEQ NOT 100 LENGTH: ", i, len(network_input[i]))

    for i in range(len(network_output)):
        if len(network_output[i]) != 100:
            print("SEQ NOT 100 LENGTH: ", i, len(network_output[i]))

    n_patterns = len(network_input)

    cutoff = int(n_patterns * .8)
    test_x = network_input[cutoff:]
    test_y = network_output[cutoff:]
    network_input = network_input[:cutoff]
    network_output = network_output[:cutoff]
    print(len(network_input))
    print(len(test_x))
    print(cutoff)
    print(n_patterns)

    print(network_input[0])
    print(network_input[0].shape)
    # one hot encode notes to make 1 x 100 x 358 tensor?

    # reshape the input into a format compatible with LSTM layers
    #network_input = torch.Tensor(numpy.reshape(network_input, (len(network_input), sequence_length, -1)))
    network_input = torch.Tensor(numpy.stack(network_input))
    print(network_input.shape)  #100 x 358 x 45660

    # normalize input
    # network_input = network_input / float(n_vocab)

    # network_output = torch.Tensor(numpy.reshape(network_output, (len(network_input), sequence_length, 358)))
    network_output = torch.Tensor(numpy.stack(network_output))
    print(network_output.shape)  #100 x 358 x 45660

    # test_x = torch.Tensor(numpy.reshape(test_x, (n_patterns - cutoff, sequence_length, 358)))
    test_x = torch.Tensor(numpy.stack(test_x))

    # normalize input
    # test_x = test_x / float(n_vocab)

    # test_y = torch.Tensor(numpy.reshape(test_y, (n_patterns - cutoff, sequence_length, 358)))
    test_y = torch.Tensor(numpy.stack(test_y))
    # normalize output
    # test_y = test_y / float(n_vocab)

    return (network_input, network_output, test_x, test_y)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    # model.load_weights('weights-improvement-195-0.1490-bigger.hdf5')
    model.load_weights('weights-improvement-31-3.0448-bigger.hdf5')


    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        if note_index % 10 == 0:
            print("on index: ", note_index)
        prediction_input = pattern

        # prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        # prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        prediction = undo_categorical(prediction)
        print("predicion shape after categorical: ", prediction.shape)
        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()
