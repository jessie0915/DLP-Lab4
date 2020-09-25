from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2
teacher_forcing_ratio = 0.8

class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.char2index = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.char2count = {}
        self.index2char = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.n_chars = 4  # Count SOS and EOS

    def addWord(self, word):
        for char in self.split_sequence(word):
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

    def split_sequence(self, word):
        """Vary from languages and tasks. In our task, we simply return chars in given sentence
        For example:
            Input : alphabet
            Return: [a, l, p, h, a, b, e, t]
        """
        return [char for char in word]

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        """Transform a char sequence to index sequence
            :param sequence: a string composed with chars
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.char2index['SOS']] if add_sos else []

        for char in self.split_sequence(sequence):
            if char not in self.char2index:
                index_sequence.append((self.char2index['UNK']))
            else:
                index_sequence.append(self.char2index[char])

        if add_eos:
            index_sequence.append(self.char2index['EOS'])

        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = ""
        for idx in indices:
            char = self.index2char[idx]
            if char == "EOS":
                break
            else:
                sequence += char
        return sequence


def readWords(data_path):
    print("Reading data...")

    max_len = 0
    pairs = []
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
        for p in data:
            words = p['input']
            target = p['target']
            w_num = len(words)
            for i in range(w_num):
                pairs.append([words[i], target])
                if max_len < len(words[i]):
                    max_len = len(words[i])
                if max_len < len(target):
                    max_len = len(target)

    data_vocab = Vocabulary('input_target')

    return data_vocab, pairs, max_len


def prepareData(path):

    data_vocab, pairs, max_len = readWords(path)
    print("Read %s word pairs" % len(pairs))
    print("Counting chars...")
    for pair in pairs:
        data_vocab.addWord(pair[0])
        data_vocab.addWord(pair[1])
    print("Counted chars:")
    print(data_vocab.name, data_vocab.n_chars, data_vocab.char2index)

    return data_vocab, pairs, max_len


data_vocab, pairs, MAX_LENGTH = prepareData('train.json')
MAX_LENGTH = MAX_LENGTH + 1
print(random.choice(pairs))
_, test_pairs, _ = prepareData('test.json')


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        nn.init.normal_(self.embedding.weight, 0.0, 0.2)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input = [input len, batch size]

        embedded = self.dropout(self.embedding(input)).view(1, 1, -1)
        # embedded = [input len, batch size, emb dim]

        # hidden = (hidden, cell)
        output, hidden = self.lstm(embedded, hidden)

        return output, hidden

    def initHidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)
        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)

        hidden = (Variable(nn.Parameter(h0, requires_grad=True)).to(device),
                  Variable(nn.Parameter(c0, requires_grad=True)).to(device))

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embedding_size)
        nn.init.normal_(self.embedding.weight, 0.0, 0.2)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        nn.init.normal_(self.out.weight, 0.0, 0.2)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        # input = [1, batch size]
        output = self.dropout(self.embedding(input)).view(1, 1, -1)
        # embedded = [1, batch size, emb dim]

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def initHidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)
        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)

        hidden = (Variable(nn.Parameter(h0, requires_grad=True)).to(device),
                  Variable(nn.Parameter(c0, requires_grad=True)).to(device))

        return hidden


def tensorFromSentence(vocab, word):
    char_sequence = vocab.split_sequence(word)
    indexes = vocab.sequence_to_indices(char_sequence, add_eos=True)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(data_vocab, pair[0])
    target_tensor = tensorFromSentence(data_vocab, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    plot_bleu_score = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    pairs_random = pairs
    random.shuffle(pairs_random)
    training_pairs = []
    for k in range(n_iters):
        training_pairs.append(tensorsFromPair(pairs_random[k % len(pairs_random)]))

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            # bleu_score_train = evaluate_all(encoder, decoder, pairs, mode='train')
            bleu_score_test = evaluate_all(encoder, decoder, test_pairs, mode='train')
            plot_bleu_score.append(bleu_score_test)

            print('%s (%d %d%%): loss=%.4f, test_bleu_score=%.4f'
                  % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg,
                     bleu_score_test))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        torch.save(encoder.state_dict(), 'encoder.dict')
        torch.save(decoder.state_dict(), 'decoder.dict')

    showPlot(plot_losses, 'loss.png')
    showPlot(plot_bleu_score, 'bleu.png')


def showPlot(points, path):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    if path == 'loss.png':
        plt.plot(points, color="blue", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("loss value")
        plt.title("Loss Curve")
    else:
        plt.plot(points, color="red", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("score")
        plt.title("BLEU Score Curve")
        plt.ylim(0.0, 1.0)

    plt.legend()
    plt.savefig(path)


def evaluate(encoder, decoder, word, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(data_vocab, word)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(data_vocab.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('input: ', pair[0])
        print('target:', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_word=''
        for k in range(len(output_words)-1):
            output_word += str(output_words[k])
        print('pred:  ', output_word)
        print('')


# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


def evaluate_all(encoder, decoder, _pairs, mode='test'):
    bleu_score = 0.0
    if mode == 'test':
        print('========================')
    for i in range(len(_pairs)):
        pair = _pairs[i]
        if mode == 'test':
            print('input: ', pair[0])
            print('target:', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_word = ''
        for k in range(len(output_words) - 1):
            output_word += str(output_words[k])
        if mode == 'test':
            print('pred:  ', output_word)
            print('========================')

        bleu_score += compute_bleu(output_word, pair[1])

    bleu_score /= len(_pairs)
    return bleu_score

def load_modle_and_evaluate(_pairs, mode='test'):
    encoder = EncoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)
    decoder = DecoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)

    encoder.load_state_dict(torch.load('encoder.dict'))
    decoder.load_state_dict(torch.load('decoder.dict'))

    return evaluate_all(encoder, decoder, _pairs, mode)

def load_modle_and_train():
    encoder = EncoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)
    decoder = DecoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)

    encoder.load_state_dict(torch.load('encoder.dict'))
    decoder.load_state_dict(torch.load('decoder.dict'))

    trainIters(encoder, decoder, 904705, print_every=5000, learning_rate=LR)


embedding_size = 512
hidden_size = 512
LR=0.05
encoder1 = EncoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)
decoder1 = DecoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)
trainIters(encoder1, decoder1, 904705, print_every=100, learning_rate=LR)

# print(load_modle_and_evaluate(test_pairs, mode='test'))



