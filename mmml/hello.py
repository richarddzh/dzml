class Hello:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def say(name):
        print('Hello {0}'.format(name))

    def say_hi(self):
        Hello.say(self.name)
