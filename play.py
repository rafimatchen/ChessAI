import chess
import numpy as np
from MCTS import MCTS
from Game import Game
from NetworkWrapper import NetworkWrapper
from PlayWrapper import PlayWrapper

game = Game()
n1 = NetworkWrapper(game)
n1.load_checkpoint('./pretrained_models/chess/keras/','8x8_100checkpoints_best.pth.tar')

mcts1 = MCTS(game, n1, {'numSims': 50, 'cpuct': 1.0})
n1p = lambda x: np.argmax(mcts1.getActionProbability(x, temp=0))

n2 = NetworkWrapper(game)
n2.load_checkpoint('./pretrained_models/chess/keras/','8x8_100checkpoints_best.pth.tar')

mcts2 = MCTS(game, n2, {'numSims': 50, 'cpuct': 1.0})
n2p = lambda x: np.argmax(mcts2.getActionProbability(x, temp=0))

player2 = n2p

playwrapper = PlayWrapper(n1p, n2p, game)

print(playwrapper.playMany(2))