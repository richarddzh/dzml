import mmml.helpers as helpers
import tensorflow as tf


char_map = helpers.file_read_all_text(r'dataset\dict.txt', 'utf-8')
text = helpers.file_read_all_text(r'dataset\baiyexing.txt', 'utf-8')
data = helpers.map_text_to_integers(text, char_map)
batch_size = 2
num_step = 4

with tf.Graph().as_default():
    x, y = helpers.sequence_input_producer(data, batch_size, num_step)
    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(2):
            print(i)
            a = session.run((x, y))
            print(a)

