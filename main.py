from __future__ import print_function

import os
import sys
import time
import random
random.seed(0)

import numpy as np
import dynet_config
dynet_config.set_gpu()
dynet_config.set(mem=1024, \
        random_seed=random.randint(1, 100),
        weight_decay=0.001
    )
import dynet as dy

def ptb(section='test.txt', directory='ptb/', padding='<EOS>', column=0):
    with open(os.path.join(directory, section), 'rt') as fh:
        data = [ i.split('\t')[column] for i in fh ]
    data = [ ex.strip().split(' ') for ex in data ]
    data = [ ex + [padding] for ex in data ]
    vocab = set([ word for sent in data for word in sent ])
    return vocab, data

def read_vocab(vocab='vocab', directory='data/'):
    with open(os.path.join(directory, vocab), 'rt') as fh:
        vocab = [ i.strip().split('\t')[0] for i in fh ]
    return vocab

def text_to_sequence(texts, vocab):
    word_to_n = { word : i for i, word in enumerate(vocab, 0) }
    n_to_word = { i : word for word, i in word_to_n.items() }
    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])
    return sequences, word_to_n, n_to_word

if __name__ == '__main__':
    print('Reading vocab...')
    in_vocab = read_vocab()
    in_vocab +=  ['<unk>', '<EOS>', '<mask>']
    out_vocab = ['(', ')', '<TOK>', '<EOS>']
    print('Done.')

    print('Reading train/valid data...')
    BATCH_SIZE = 128
    _, X_train = ptb(section='wsj_2-21', directory='data/', column=0)
    _, y_train = ptb(section='wsj_2-21', directory='data/', column=1)
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train, in_vocab)
    y_train_seq, _, _ = text_to_sequence(y_train, out_vocab)

    _, X_valid = ptb(section='wsj_24', directory='data/', column=0)
    _, y_valid = ptb(section='wsj_24', directory='data/', column=1)
    X_valid_seq, word_to_n, _ = text_to_sequence(X_valid, in_vocab)
    y_valid_seq, _, _ = text_to_sequence(y_valid, out_vocab)
    print('Done.')

    print('Contains %d unique words.' % len(in_vocab))
    print('Read in %d examples.' % len(X_train))

    print('Checkpointing models on validation loss...')
    lowest_val_loss = 0.

    RUN = 'runs/baseline'
    checkpoint = os.path.join(RUN, 'baseline.model')
    print('Checkpoints will be written to %s.' % checkpoint)

    print('Building model...')
    collection = dy.ParameterCollection()
    print('Done.')

    print('Training model...')
    EPOCHS = 1000
    trainer = dy.SimpleSGDTrainer(collection, learning_rate=0.4)

    for epoch in range(1, EPOCHS+1):
        loss = 0.
        start = time.time()

        for i, (X_batch, y_batch, X_masks, y_masks) in \
                enumerate(zip(X_train_seq, y_train_seq, X_train_masks, y_train_masks), 1):
            dy.renew_cg()
            batch_loss, _ = seq2seq.one_batch(X_batch, y_batch, X_masks, y_masks)
            batch_loss.backward()
            trainer.update()

            elapsed = time.time() - start
            loss += batch_loss.value()
            avg_batch_loss = loss / i
            ex = min(len(X_train), i * BATCH_SIZE)

            print('Epoch %d. Time elapsed: %ds, %d/%d. Average batch loss: %f\r' % \
                    (epoch, elapsed, ex, len(X_train), avg_batch_loss), end='')

        print()
        print('Done. Total loss: %f' % loss)
        trainer.status()
        print()

        trainer.learning_rate *= 0.98

        print('Validating...')
        loss = 0.
        correct_toks = 0.
        total_toks = 0.

        validation = open(os.path.join(RUN, 'validation'), 'wt')
        for i, (X_batch, y_batch, X_masks, y_masks, X_batch_raw, y_batch_raw) in \
                enumerate(zip(X_valid_seq, y_valid_seq, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw), 1):
            dy.renew_cg()
            batch_loss, decoding = seq2seq.one_batch(X_batch, y_batch, X_masks, y_masks, training=False)
            loss += batch_loss.value()

            y_pred = seq2seq.to_sequence_batch(decoding, out_vocab)
            for X_raw, y_raw, y_ in zip(X_batch_raw, y_batch_raw, y_pred):
                validation.write('%s\t%s\t%s\n' % (' '.join(X_raw), ' '.join(y_raw), ' '.join(y_)))
                correct_toks += [ tok_ == tok for tok, tok_ in zip(y_, y_raw) ].count(True)
                total_toks += len(y_)
        validation.close()

        accuracy = correct_toks/total_toks
        print('Validation loss: %f. Token-level accuracy: %f.' % (loss, accuracy))

        if lowest_val_loss == 0. or loss < lowest_val_loss:
            print('Lowest validation loss yet. Saving model...')
            collection.save(checkpoint)
            lowest_val_loss = loss

        if highest_val_accuracy == 0. or accuracy > highest_val_accuracy:
            print('Highest accuracy yet. Saving model...')
            collection.save(checkpoint)
            highest_val_accuracy = accuracy
        print('Done.')

    print('Done.')
