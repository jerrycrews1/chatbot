from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense, Input
import numpy as np

# import wandb
# from wandb.keras import WandbCallback
#
# wandb.init()
# config = wandb.config

batch_size = 1000
epochs = 10000
latent_dim = 256
num_samples = 200000  # this needs to be smaller on computers with less than 16gb memory.
unclean_input = 'train.from'
unclean_output = 'train.to'

input_texts = []
target_texts = []
input_words = set()
target_words = set()

# input
with open(unclean_input, 'r', encoding="utf-8") as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_texts.append(line)
    for word in line.split():
        if word not in input_words:
            input_words.add(word)
# target
with open(unclean_output, 'r', encoding="utf-8") as of:
    output_lines = of.read().split('\n')
for line in output_lines[: min(num_samples, len(lines) - 1)]:
    target_texts.append(line)
    for word in line.split():
        if word not in target_words:
            target_words.add(word)

input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
max_encoder_seq_length = 50
max_decoder_seq_length = 50

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

