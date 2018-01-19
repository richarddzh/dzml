import tflearn
import matplotlib.pyplot as plt
import numpy as np

input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(
    linear,
    optimizer='sgd',
    loss='mean_square',
    metric='R2',
    learning_rate=0.01)

m = tflearn.DNN(regression)

# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("Regression result: Y = {0}*X + {1}".format(
    m.get_weights(linear.W),
    m.get_weights(linear.b)))

X1 = np.linspace(0, 12, 20)
Y1 = X1 * m.get_weights(linear.W) + m.get_weights(linear.b)
fig, ax = plt.subplots()
ax.plot(X, Y, 'r+')
ax.plot(X1, Y1, '--')
plt.show()