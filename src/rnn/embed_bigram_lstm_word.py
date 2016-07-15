#-*- coding: utf-8 -*-
# pylint: disable=C,W
import random
import string
import zipfile

import numpy as np
import tensorflow as tf

from not_mnist.img_pickle import save_obj, load_pickle
from not_mnist.load_data import maybe_download


def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


data_set = load_pickle('text8_text.pickle')
if data_set is None:
    # load data
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', 31344016, url=url)

    # read data
    text = read_data(filename)
    print('Data size %d' % len(text))
    save_obj('text8_text.pickle', text)
else:
    text = data_set

# use word instead of characters
text = text.split(' ')

# Create a small validation set.
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

# Utility functions to map words to vocabulary IDs and back.
vocabulary = {}
for word in text:
    vocabulary[word] = vocabulary.get(word, 0) + 1
vocabulary = filter(lambda (word, cnt): cnt > 10, vocabulary.iteritems())
vocabulary = map(lambda (word, cnt): word, vocabulary)
if '<unk>' in vocabulary:
    vocabulary.remove('<unk>')
vocabulary.insert(0, '<unk>')
vocabulary = list(enumerate(vocabulary))
index_word = dict(vocabulary)
word_index = dict(map(lambda (ind, word): (word, ind), vocabulary))
vocabulary_size = len(vocabulary)

print '==================== vocabulary_size is %s =================' % vocabulary_size
print(word_index.get('this'), word_index.get('z'), word_index.get(' '), word_index.get('ï'), word_index.get('<unk>'))
print(index_word.get(1), index_word.get(26), index_word.get(0))



class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        batch = np.zeros(shape=self._batch_size, dtype=np.int)
        # print 'batch idx %i' %
        for b in range(self._batch_size):
            word_idx = self._cursor[b]
            word = word_index.get(self._text[word_idx], 0)
            batch[b] = word
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def batch2string(encodings):
    return [index_word.get(e, '<unk>') for e in encodings]


def batches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [' '.join(x) for x in zip(s, batch2string(b))]
    return s


train_batches = BatchGenerator(train_text, 8, 8)
valid_batches = BatchGenerator(valid_text, 1, 1)

batch = train_batches.next()
print(batch)
print(batches2string(batch))
# print onehot(batch)
print (batches2string(train_batches.next()))
print (batches2string(valid_batches.next()))
print (batches2string(valid_batches.next()))


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction, size=vocabulary_size):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def one_hot_voc(prediction, size=vocabulary_size):
    p = np.zeros(shape=[1, size], dtype=np.float)
    p[0, prediction[0]] = 1.0
    return p


def onehot(encodings, size=vocabulary_size):
    res = []
    for e in encodings:
        p = np.zeros(shape=[1, size], dtype=np.float)
        p[0, e] = 1.0
        res.append(p)
    return res


def random_distribution(size=vocabulary_size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, size])
    return b / np.sum(b, 1)[:, None]


def create_lstm_graph(num_nodes, num_unrollings, batch_size, embedding_size=vocabulary_size):
    with tf.Graph().as_default() as g:
        # input to all gates
        x = tf.Variable(tf.truncated_normal([embedding_size, num_nodes * 4], -0.1, 0.1), name='x')
        # memory of all gates
        m = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], -0.1, 0.1), name='m')
        # biases all gates
        biases = tf.Variable(tf.zeros([1, num_nodes * 4]))
        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([vocabulary_size]))
        # embeddings for all possible words
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
        # one hot encoding for labels in
        #np_embeds = np.zeros((vocabulary_size, vocabulary_size))
        #np.fill_diagonal(np_embeds, 1)
        #onehot = tf.constant(np.reshape(np_embeds, -1), dtype=tf.float32, shape=[vocabulary_size, vocabulary_size],
        #                           name='onehot')



        tf_keep_prob = tf.placeholder(tf.float32, name='tf_keep_prob')

        # Definition of the cell computation.
        def lstm_cell(i, o, state):
            # apply dropout to the input
            i = tf.nn.dropout(i, tf_keep_prob)
            mult = tf.matmul(i, x) + tf.matmul(o, m) + biases
            input_gate = tf.sigmoid(mult[:, :num_nodes])
            forget_gate = tf.sigmoid(mult[:, num_nodes:num_nodes * 2])
            update = mult[:, num_nodes * 3:num_nodes * 4]
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(mult[:, num_nodes * 3:])
            output = tf.nn.dropout(output_gate * tf.tanh(state), tf_keep_prob)
            return output, state

        # Input data. [num_unrollings, batch_size] -> one hot encoding removed, we send just word ids
        tf_train_data = tf.placeholder(tf.int32, shape=[num_unrollings + 1, batch_size], name='tf_train_data')
        train_data = list()
        for i in tf.split(0, num_unrollings + 1, tf_train_data):
            train_data.append(tf.squeeze(i))
        train_inputs = train_data[:num_unrollings]
        train_labels = list()
        #train_labels = train_data[1:]
        for l in train_data[1:]:
            # train_labels.append(tf.nn.embedding_lookup(embeddings, l))
            #train_labels.append(tf.gather(onehot, l))
            train_labels.append(tf.one_hot(l, vocabulary_size))
            # train_labels.append(tf.reshape(l, [batch_size,1]))  # labels are inputs shifted by one time step.

        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        # python loop used: tensorflow does not support sequential operations yet
        for i in train_inputs:  # having a loop simulates having time
            # embed input words -> [batch_size, embedding_size]
            output, state = lstm_cell(tf.nn.embedding_lookup(embeddings, i), output, state)
            outputs.append(output)

        # State saving across unrollings, control_dependencies makes sure that output and state are computed
        with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
            logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                          tf.concat(0, train_labels)
                                                                          ), name='loss')
            #logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            #num_sampled = 64
            #loss = tf.reduce_mean(
            #    tf.nn.sampled_softmax_loss(w, b, tf.concat(0, outputs), train_labels,
            #                              num_sampled, vocabulary_size))
        # Optimizer.
        global_step = tf.Variable(0, name='global_step')
        learning_rate = tf.train.exponential_decay(10.0, global_step, 500, 0.9, staircase=True, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer')
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # here we predict the embedding
        # train_prediction = tf.argmax(tf.nn.softmax(logits), 1, name='train_prediction')
        train_prediction = tf.nn.softmax(logits, name='train_prediction')

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.int32, shape=[1], name='sample_input')
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]), name='saved_sample_output')
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]), name='saved_sample_state')
        reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),
                                      saved_sample_state.assign(tf.zeros([1, num_nodes])), name='reset_sample_state')
        embed_sample_input = tf.nn.embedding_lookup(embeddings, sample_input)
        sample_output, sample_state = lstm_cell(embed_sample_input, saved_sample_output, saved_sample_state)

        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b), name='sample_prediction')
        return g


# test graph
create_lstm_graph(64, 10, 128, 32)


def train(g, num_steps, summary_frequency, num_unrollings, batch_size):
    # initalize batch generators
    train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)
    optimizer = g.get_tensor_by_name('optimizer:0')
    loss = g.get_tensor_by_name('loss:0')
    train_prediction = g.get_tensor_by_name('train_prediction:0')
    learning_rate = g.get_tensor_by_name('learning_rate:0')
    tf_train_data = g.get_tensor_by_name('tf_train_data:0')
    sample_prediction = g.get_tensor_by_name('sample_prediction:0')
    # similarity = g.get_tensor_by_name('similarity:0')
    reset_sample_state = g.get_operation_by_name('reset_sample_state')
    sample_input = g.get_tensor_by_name('sample_input:0')
    embeddings = g.get_tensor_by_name('embeddings:0')
    keep_prob = g.get_tensor_by_name('tf_keep_prob:0')
    with tf.Session(graph=g) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        mean_loss = 0
        for step in range(num_steps):
            batches = train_batches.next()
            # print batches2string(batches)
            # print np.array(batches)
            # feed_dict = dict()
            # for i in range(num_unrollings + 1):
            #  feed_dict[train_data[i]] = batches[i]
            # tf_train_data =
            _, l, lr, predictions = session.run([optimizer, loss, learning_rate, train_prediction],
                                                feed_dict={tf_train_data: batches, keep_prob: 0.6})
            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print ('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0
                labels = list(batches)[1:]
                print predictions
                labels = np.concatenate([onehot(l) for l in labels])
                print labels
                print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels)/len(labels))))
                #print('Minibatch perplexity: %.2f' % float(logprob(predictions, labels)))
                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    # print embeddings.eval()
                    for _ in range(5):
                        # print random_distribution(vocabulary_size)
                        feed = np.argmax(sample(random_distribution(vocabulary_size), vocabulary_size))
                        sentence = index_word[feed]
                        reset_sample_state.run()
                        for _ in range(49):
                            # prediction = similarity.eval({sample_input: [feed]})
                            # nearest = (-prediction[0]).argsort()[0]
                            prediction = sample_prediction.eval({sample_input: [feed], keep_prob: 1.0})
                            # print prediction
                            feed = np.argmax(sample(prediction, vocabulary_size))
                            # feed = np.argmax(prediction[0])
                            sentence += ' ' +  index_word[feed]
                        print(sentence)
                    print('=' * 80)
                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_size):
                    b = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: b[0], keep_prob: 1.0})
                    # print(predictions)
                    valid_logprob = valid_logprob + logprob(predictions, one_hot_voc(b[1], vocabulary_size))
                print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))


num_unrollings = 5
batch_size=16
num_nodes = 128
graph = create_lstm_graph(num_nodes, num_unrollings, batch_size, 128)
train(graph, 4001, 1, num_unrollings, batch_size)
