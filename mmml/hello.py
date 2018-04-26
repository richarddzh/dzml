import tensorflow as tf


class Hello:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def say(name):
        print('Hello {0}'.format(name))

    def say_hi(self):
        Hello.say(self.name)

    @staticmethod
    def use_range_input_producer():
        with tf.Graph().as_default():
            i = tf.train.range_input_producer(4).dequeue()
            # Usage 1
            with tf.train.MonitoredTrainingSession() as session:
                for j in range(10):
                    a = session.run(i)
                    print(a)
            # Usage 2
            sv = tf.train.Supervisor()
            with sv.managed_session() as session:
                for j in range(10):
                    a = session.run(i)
                    print(a)
