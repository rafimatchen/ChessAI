import time
import random
import numpy as np 
import math
from network import Network
import os

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
}


class NetworkWrapper():
    def __init__(self, game):
        self.network = Network(game, args)
        self.actionSize = game.getActionSize()
        self.game = game

    def train(self, examples):
        inputBoards, targetPis, targetVis = list(zip(*examples))
        inputBoards = np.asarray(inputBoards)
        targetPis = np.asarray(targetPis)
        targetVis = np.asarray(targetVis)
        self.network.model.fit(inputBoards, [targetPis, targetVis], batch_size = 128, epochs=100)

    def predict(self, board):
        start = time.time()

        board = np.array([hash(self.game.stringRepresentation(board))])
        pi, v = self.network.model.predict(board)
        
        return pi[0], v[0]

    def saveCheckpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.network.model.saveWeights(filepath)

    def loadCheckpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise('No model in path')
        self.network.model.loadWeihts(filepath)
