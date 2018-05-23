import mmml.helpers as helpers
import mmml.lstmmodel as lstmmodel
import tensorflow as tf


char_map = helpers.file_read_all_text(r'dataset\dict.txt', 'utf-8')
text = helpers.file_read_all_text(r'dataset\baiyexing.txt', 'utf-8')
data = helpers.map_text_to_integers(text, char_map)
batch_size = 2
num_step = 4
vocab_size = len(char_map)
embed_size = 1024
hidden_size = 128
num_layers = 2
learn_rate = 1.0
init_scale = 0.1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.variable_scope("model", initializer=initializer):
        x, y = helpers.sequence_input_producer(data, batch_size, num_step)
        x = helpers.make_embedding(x, vocab_size, embed_size)
        m = lstmmodel.LSTMModel(hidden_size, num_layers, reuse=False)
        m.compute(batch_size, x)
        x = tf.reshape(m.outputs, [-1, hidden_size])
        x = helpers.make_dense(x, 'dense', hidden_size, vocab_size)
        x = tf.reshape(x, [batch_size, num_step, vocab_size])
        loss = tf.contrib.seq2seq.sequence_loss(
            x, y,
            tf.ones([batch_size, num_step], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        cost = tf.reduce_sum(loss)
        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost, global_step)
    with tf.train.MonitoredTrainingSession(checkpoint_dir='./checkpoints/chinese/') as session:
        fetches = {
            "train_op": train_op,
            "cost": cost,
            "global_step": global_step,
        }
        for i in range(10000):
            values = m.run(session, {}, fetches)
            print(values["global_step"], values["cost"])
