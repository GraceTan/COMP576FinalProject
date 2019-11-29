""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes() # a long list of notes ex. 'D2', '11.4', 'G#4', 'F#1'

    # get amount of pitch names
    n_vocab = len(set(notes))   # how many unique notes there are
    print("n_vocab: ", n_vocab)

    network_input, network_output, test_x, test_y = prepare_sequences(notes, n_vocab)

    s2s, optimizer, criterion = create_network(network_input, n_vocab)
    print("made model")

    train(s2s, optimizer, criterion, network_input, network_output, test_x, test_y)
    print("done training")

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

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


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

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


    
    # normalize output
    # network_output = network_output / float(n_vocab)

    # network_output = to_categorical(network_output)

    # test_x = torch.Tensor(numpy.reshape(test_x, (n_patterns - cutoff, sequence_length, 358)))
    test_x = torch.Tensor(numpy.stack(test_x))

    # normalize input
    # test_x = test_x / float(n_vocab)

    # test_y = torch.Tensor(numpy.reshape(test_y, (n_patterns - cutoff, sequence_length, 358)))
    test_y = torch.Tensor(numpy.stack(test_y))
    # normalize output
    # test_y = test_y / float(n_vocab)

    return (network_input, network_output, test_x, test_y)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers = 2, dropout = .5)
        
    def forward(self, input):
        output, (hidden, cell) = self.gru(input)
        return hidden, cell
    
    def initHidden(self):
        # 45660 is number of training lists (of len 100)
        return torch.zeros(2, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size, num_layers = 2, dropout = .5)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, hidden, cell):
        # print("decoder forward")
        output = F.relu(input)
        output, (hidden, cell) = self.gru(output, torch.stack((hidden, cell), dim = 0))
        output = self.sigmoid(self.out(output[0]))
        return output, hidden, cell
    
    def initHidden(self):
        return torch.zeros(2,1, self.hidden_size)

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        # self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # print("src: ", src.shape)
        # print("target: ", trg.shape)
        #initialize output vector
        # outputs = torch.zeros((102,100))
        # outputs = torch.zeros((100, 358))
        outputs = torch.zeros((100, 358))

        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden, cell = self.encoder(src.view(102,1,219))
        hidden, cell = self.encoder(src.view(100, 1, 358))

        # print("hidden shape; ", hidden.shape)
        # print("cell: ", cell.shape)
        
        # input = trg[0,:] #SOS token
        input = trg[0, :]
        # print("input; ", input)
            
        for t in range(0, 100):
            # output, hidden, cell = self.decoder(input.view(1,1,219), hidden, cell)
            output, hidden, cell = self.decoder(input.view(1, -1, 358), hidden, cell)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = (trg[t] if teacher_force else output)
        
        return outputs


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    encoder = EncoderRNN(358, 512) # first param = 100?  
    decoder = DecoderRNN(512, 358)
    s2s = seq2seq(encoder, decoder)
    optimizer = optim.Adam(s2s.parameters())
    criterion = nn.BCELoss()

    return s2s, optimizer, criterion

def train(s2s, optimizer, criterion, network_input, network_output, test_x, test_y):
    """ train the neural network """
    # filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    # checkpoint = ModelCheckpoint(
    #     filepath,
    #     monitor='loss',
    #     verbose=0,
    #     save_best_only=True,
    #     mode='min'
    # )
    # callbacks_list = [checkpoint]

    # model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

    num_epochs = 20
    
    tr_loss_list = []
    tt_loss_list = []
    
    for epoch in range(num_epochs):
        
        #TRAINING
        s2s.train()
        epoch_tr_loss = 0
        print("length of network_input ", len(network_input))
        for i in range(200): 
            if i == 0:
                print("network_input[i] shape: ", network_input[i].shape)
            optimizer.zero_grad()
            output = s2s(network_input[i], network_output[i])
            loss = criterion(output, network_output[i])
            
            loss.backward()
            
            optimizer.step()
            
            epoch_tr_loss += loss.item()
            
            if i % 100 == 0:
                print("Training iteration ", i, " out of ", len(network_input))
            
            
        #TESTING
        s2s.eval()
        epoch_tt_loss = 0
        # for i in range(len(test_x)): 
        for i in range(200): 
            output = s2s(test_x[i], test_y[i], 0)
            
            loss = criterion(output, test_y[i])
            
            epoch_tt_loss += loss.item()
            
            if i % 100 == 0:
                print("Testing iteration ", i, " out of ", len(test_x))
            
            
        tr_loss_list.append(epoch_tr_loss/len(network_input))
        tt_loss_list.append(epoch_tt_loss/len(test_x))
    
        print('We are on epoch ', epoch)
        print('The current training loss is ', epoch_tr_loss)
        print('The current test loss is ', epoch_tt_loss)
        print()
        
        if epoch % 10 == 0:
            state = {
                    'epoch': epoch,
                    'state_dict': s2s.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }
            
            torch.save(state, 'mymodelstate.pth')
            

if __name__ == '__main__':
    train_network()
