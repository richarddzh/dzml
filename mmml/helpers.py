import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import re
import os.path as path


def file_read_all_text(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as file:
        text = file.read()
    return text


def file_write_all_text(file_path, text, encoding):
    with open(file_path, 'w', encoding=encoding) as file:
        file.write(text)


def clean_text_file(in_file_path, out_file_path, encoding):
    text = file_read_all_text(in_file_path, encoding)
    text = re.sub(r'\s+', ' ', text)
    file_write_all_text(out_file_path, text, encoding)


def create_character_map(map_file_path, in_file_path, encoding):
    character_map = ''
    if path.exists(map_file_path):
        character_map = file_read_all_text(map_file_path, encoding)
    text = file_read_all_text(in_file_path, encoding)
    for c in text:
        if c not in character_map:
            character_map = character_map + c
    character_map = ''.join(sorted(set(character_map)))
    file_write_all_text(map_file_path, character_map, encoding)


def map_text_to_integers(text, character_map):
    return [character_map.index(c) for c in text]


def map_integers_to_text(integers, character_map):
    character_array = [character_map[i] for i in integers]
    return ''.join(character_array)


def file_read_text_as_integers(text_file_path, map_file_path, encoding):
    text = file_read_all_text(text_file_path, encoding)
    character_map = file_read_all_text(map_file_path, encoding)
    return map_text_to_integers(text, character_map)


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


def make_lstm(inputs, initial_state, hidden_size, num_layers, reuse):
    def make_cell():
        return rnn.BasicLSTMCell(hidden_size, reuse=reuse)
    cell = rnn.MultiRNNCell([make_cell() for _ in range(num_layers)])
    outputs, state = rnn.static_rnn(cell, inputs, initial_state=initial_state)
    return outputs, state
