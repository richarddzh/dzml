import mmml.helpers as helpers
import tensorflow as tf


char_map = helpers.file_read_all_text(r'dataset\ZiDian.txt', 'utf-8')
text = helpers.file_read_all_text(r'dataset\BaiYeXing.txt', 'utf-8')
data = helpers.map_text_to_integers(text, char_map)
batch_size = 2
num_step = 4

with tf.Graph().as_default():
    x, y = helpers.sequence_input_producer(data, batch_size, num_step)

