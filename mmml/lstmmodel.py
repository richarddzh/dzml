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
        self.state_value = None

    def compute(self, batch_size, inputs):
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        inputs = tf.unstack(inputs, axis=1)
        outputs, state = rnn.static_rnn(self.cell, inputs, initial_state=self.initial_state)
        self.outputs = tf.stack(outputs, axis=1)
        self.final_state = state

    def run(self, session, feed_dict, fetches):
        if self.state_value is None:
            self.state_value = session.run(self.initial_state)
        fetches["final_state"] = self.final_state
        feed_dict[self.initial_state] = self.state_value
        values = session.run(fetches, feed_dict)
        self.state_value = values["final_state"]
        return values

    def _make_cell(self, reuse):
        return rnn.BasicLSTMCell(self.hidden_size, reuse=reuse)

    def _make_multi_cell(self, reuse):
        return rnn.MultiRNNCell([self._make_cell(reuse) for _ in range(self.num_layers)])
