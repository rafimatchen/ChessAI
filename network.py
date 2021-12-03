import sys
sys.path.append('..')
import keras

class Network():
    def __init__(self, game, args):
        self.actionSize = game.getActionSize()

        self.inputBoards = keras.layers.Input(shape=(1))

        x = keras.layers.Reshape((1, 1, 1))(self.inputBoards)
        x = keras.layers.BatchNormalization(axis=2)(keras.layers.Conv1D(16, 3, padding='same')(x))
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(args['dropout'])(keras.layers.BatchNormalization(axis=1)(keras.layers.Dense(1024, use_bias=False)(x)))  # batch_size x 1024
        x = keras.layers.Dropout(args['dropout'])(keras.layers.BatchNormalization(axis=1)(keras.layers.Dense(512, use_bias=False)(x)))          # batch_size x 1024
        self.pi = keras.layers.Dense(self.actionSize, activation='softmax', name='pi')(x)   # batch_size x self.action_size
        self.v = keras.layers.Dense(1, activation='tanh', name='v')(x)                    # batch_size x 1

        self.model = keras.Model(inputs=self.inputBoards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer='adam')
        print(self.model.summary())