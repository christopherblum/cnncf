# cnncf
Convolutional Neural network with circular filters for sequence motif inference ([bioRxiv preprint](https://doi.org/10.1101/312959))


## Description

This python script demonstrates how circular filters can be applied to sequence motif inference with convolutional neural networks.

First, a data set of `num_positive` positive and `num_negative` negative training sequences is created, each sequence of length `sequence_length`. The positive training sequences all contain a motif `motif` placed at a random position in a random sequence background. Negative sequences are created by randomly shuffling the nucleotide order of all positive training sequences. Motifs can only contain letters A, C, G or T (upper-case).

Two models are trained: a CNN with circular filters and a regular CNN. Both use filters of length `filter_length`.
The models are trained for `training_steps` training steps at a learning rate of `learning_rate` with mini-batches containing `batch_size` examples. Regularization strength can be adjusted with `regul`, and models can be trained on GPUs (although not necessary). The training accuracy can be displayed every `display_steps` training steps if not set to 0.

## Requirements: 

Numpy >= 1.14.2

TensorFlow >=3.6


## Usage:

`python cnncf.py`

## Default options:
```
--motif='ACGTAC'

--filter_length=6

--learning_rate=0.1

--sequence_length=40

--training_steps=3000

--display_steps=500

--gpus=''

--num_positive=100

--num_negative=10000

--batch_size=100

--regul=0.01
```
