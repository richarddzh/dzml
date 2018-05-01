import mmml.helpers as helpers
import tensorflow as tf


data = helpers.file_read_text_as_integers(r'dataset\HaiBianDeKaFuKa.txt', r'dataset\ZiDian.txt', 'utf8')
with tf.Graph().as_default():
    input = helpers.sequence_input_producer(data, batch_size=2, num_step=5)
    with tf.train.MonitoredTrainingSession() as session:
        a = session.run(input)
        print(a)
        