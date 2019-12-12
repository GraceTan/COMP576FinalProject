COMP 576 Fall 2019

### The project

This project aims to generate human-sounding music using a GRU-LSTM cell RNN trained on Google's MAESTRO dataset.

### The model

We implemented an RNN with both GRU and LSTM cells. The model can be found in train.py file within the rnn folder. From there, a set 
of files will be generated representing the weights. The desired weight file can then be used to iteratively predict notes during note 
generation. This code can be found in rnn/generate.py.

