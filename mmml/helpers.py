import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def create_vocabulary(text, vocabulary):
    if vocabulary is None:
        vocabulary = {}
    for c in text:
        if c not in vocabulary:
            vocabulary[c] = 0
        vocabulary[c] += 1
    return vocabulary


def make_conv2d(x, name_scope, input_depth, kernel_size, n_kernel, pool_size):
    with tf.name_scope(name_scope):
        w_size = kernel_size + [input_depth] + [n_kernel]
        w = tf.truncated_normal(w_size, stddev=0.1)
        w = tf.Variable(w)
        b_size = [n_kernel]
        b = tf.constant(0.1, shape=b_size)
        b = tf.Variable(b)
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(h + b)
        p_size = [1] + pool_size + [1]
        h = tf.nn.max_pool(h, ksize=p_size, strides=p_size, padding='SAME')
    return h


def make_dense(x, name_scope, n_input, n_output):
    with tf.name_scope(name_scope):
        w_size = [n_input, n_output]
        w = tf.truncated_normal(w_size, stddev=0.1)
        w = tf.Variable(w)
        b_size = [n_output]
        b = tf.constant(0.1, shape=b_size)
        b = tf.Variable(b)
        h = tf.matmul(x, w) + b
        h = tf.nn.relu(h)
    return h


def make_classifier(x, y, func, name_scope, learning_rate):
    with tf.name_scope(name_scope):
        y_out = func(x)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_out)
        loss = tf.reduce_mean(loss)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        accuracy = tf.equal(tf.arg_max(y_out, 1), y)
        accuracy = tf.cast(accuracy, tf.float32)
        accuracy = tf.reduce_mean(accuracy)
    return train_step, loss, accuracy


def make_lstm(x, hidden_size, num_layers):
    def make_cell(size, reuse):
        return rnn.BasicLSTMCell(size, reuse=reuse)
    cell = rnn.MultiRNNCell([])
    return cell
