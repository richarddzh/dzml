import tensorflow as tf
import tflearn

with tf.Graph().as_default():
    g = tflearn.input_data(shape=[None, 2])
    g = tflearn.fully_connected(g, 8, activation='linear')
    g = tflearn.fully_connected(g, 8, activation='relu')
    g = tflearn.fully_connected(g, 5, activation='sigmoid')
    g = tflearn.regression(g, optimizer='sgd', learning_rate=2.0, loss='mean_square')
    m = tflearn.DNN(g)
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    Y = [[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0]]
    m.fit(X, Y, n_epoch=1000, snapshot_epoch=False)
    print("Not/And/Or/Xor 0,0: {0}".format(m.predict([[0.0, 0.0]])))
    print("Not/And/Or/Xor 0,1: {0}".format(m.predict([[0.0, 1.0]])))
    print("Not/And/Or/Xor 1,0: {0}".format(m.predict([[1.0, 0.0]])))
    print("Not/And/Or/Xor 1,1: {0}".format(m.predict([[1.0, 1.0]])))
