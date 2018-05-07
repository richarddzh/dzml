import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class LSTMModel:
    def __init__(self, hidden_size, num_layers, reuse):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = self._make_cell(reuse)
        self.initial_state = None
        self.final_state = None
        self.outputs = None

    def run(self, batch_size, inputs):
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        outputs, state = rnn.static_rnn(self.cell, inputs, initial_state=self.initial_state)
        self.outputs = outputs
        self.final_state = state

    def _make_cell(self, reuse):
        return rnn.BasicLSTMCell(self.hidden_size, reuse=reuse)

    def _make_multi_cell(self, reuse):
        return rnn.MultiRNNCell([self._make_cell(reuse) for _ in range(self.num_layers)])
