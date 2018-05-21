import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist
import mmml.helpers as mh

data = mnist.input_data.read_data_sets('data')
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.int64, [None])


def classifier(x):
    x = tf.reshape(x, [-1, 28, 28, 1])
    x = mh.make_conv2d(x, 'conv1', 1, [3,3], 16, [2,2])
    x = mh.make_conv2d(x, 'conv2', 16, [3,3], 16, [2,2])
    x = tf.reshape(x, [-1, 7*7*16])
    x = mh.make_dense(x, 'dense1', 7*7*16, 64)
    x = mh.make_dense(x, 'dense2', 64, 10)
    return x


train_step, loss, accuracy = mh.make_classifier(X, Y, classifier, 0.05)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = data.train.next_batch(100)
        train_step.run(feed_dict={X: batch[0], Y: batch[1]})
        accuracy_train = accuracy.eval(feed_dict={X: batch[0], Y: batch[1]})
        accuracy_test = accuracy.eval(feed_dict={X: data.test.images, Y: data.test.labels})
        print('{0}: Train accuracy: {1}'.format(i, accuracy_train))
        print('Test accuracy: {0}'.format(accuracy_test))
