import tensorlayer as tl
import tensorflow as tf

sess = tf.InteractiveSession()
X1,Y1,X2,Y2,X3,Y3 = tl.files.load_mnist_dataset()

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.int64, shape=[None,], name="y")

net = tl.layers.InputLayer(x, name="input")
net = tl.layers.ReshapeLayer(net, shape=[-1, 28, 28, 1], name="reshape")
net = tl.layers.Conv2d(net, n_filter=32, filter_size=(5,5), act=tf.nn.relu, name="conv1")
net = tl.layers.MaxPool2d(net, filter_size=(2,2), name="pool1")
net = tl.layers.Conv2d(net, n_filter=32, filter_size=(3,3), act=tf.nn.relu, name="conv2")
net = tl.layers.MaxPool2d(net, filter_size=(2,2), name="pool2")
net = tl.layers.FlattenLayer(net, name="flatten")
net = tl.layers.DenseLayer(net, n_units=128, act=tf.nn.relu, name="dense")
net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name="softmax")
out = net.outputs
cost = tl.cost.cross_entropy(out, y, name="cost")
correct = tf.equal(tf.arg_max(out, 1), y)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

train_params = net.all_params
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)
tl.utils.fit(sess, net, train_op, cost, X1, Y1, x, y, acc,
             batch_size=50, n_epoch=1, X_val=X2, y_val=Y2, print_freq=5, eval_train=False)

tl.files.save_npz(net.all_params, name="model.npz")
sess.close()



