import tflearn
import numpy as np

X, Y, testX, testY = tflearn.datasets.mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

network = tflearn.input_data(shape=[None, 28, 28, 1])
network = tflearn.conv_2d(network, 32, 5, activation='relu', regularizer='L2')
network = tflearn.max_pool_2d(network, 2)
network = tflearn.conv_2d(network, 32, 5, activation='relu', regularizer='L2')
network = tflearn.max_pool_2d(network, 2)
network = tflearn.fully_connected(network, 128, activation='relu')
network = tflearn.fully_connected(network, 10, activation='softmax')
network = tflearn.regression(network, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

model = tflearn.DNN(network)
model.fit(X, Y, n_epoch=4,
          snapshot_epoch=True,
          batch_size=20,
          validation_set=(testX, testY),
          show_metric=True)

model.save('minst.tfl')
predictY = model.predict(testX)
predictC = np.argmax(predictY, 1)
testC = np.argmax(testY, 1)
accuracy = np.average(testC == predictC)

print("Accuracy = {0}".format(accuracy))