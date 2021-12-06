import numpy as np 
from math import sqrt

EPS = 1e-8

class MCTS():
    def __init__(self, game, network, args):
        self.game = game
        self.network = network
        self.args = args
        self.Qtable = {}
        self.Ntable = {} # times that (state, action) was visited
        self.Ns = {} # times that board was visited
        self.Ps = {} #initial policy
        self.Es = {} 
        self.Vs = {}

    def getActionProbability(self, board, temp=1):
        for i in range(self.args['numSims']):
            self.search(board)

        state = self.game.stringRepresentation(board)
        print(self.Ns)
        counts = [self.Ns[(state, action)] if (state, action) in self.Ns else 0 for action in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probabilities = [0] * len(counts)
            probabilities[bestA] = 1
            return probabilities

        counts = [x ** (1. / temp) for x in counts]
        countsSum = float(sum(counts))
        probabilities = [x / countsSum for x in counts]
        return probabilities

    def search(self, board):
        state = self.game.stringRepresentation(board)

        if state not in self.Es:
            self.Es[state] = self.game.getGameEnded(board, 1)
        if self.Es[state] != 0:
            return -self.Es[state]

        if state not in self.Ps:
            self.Ps[state], v = self.network.predict(board)
            valids = self.game.getValidMoves(board, 1)
            if self.Ps[state] not in valids:
                self.Ps[state] = 0
            sum_Ps_s = np.sum(self.Ps[state])
            if sum_Ps_s > 0:
                self.Ps[state] /= sum_Ps_s
            else:
                print('all valid moves masked')
                self.Ps[state] = self.Ps[state] + valids
                self.Ps[state] /= np.sum(self.Ps[state])

            self.Vs[state] = valids
            self.Ns[state] = 0
            return -v
        
        valids = self.Vs[state]
        best = -float('inf')
        bestAction = -1

        for action in range(self.game.getActionSize()):
            if valids[action]:
                if (state, action) in self.Qsa:
                    u = self.Qsa[(state, action)] + self.args['cpuct'] * self.Ps[state][action] * sqrt(self.Ns[state]) / (
                            1 + self.Nsa[(state, action)])
                else:
                    u = self.args['cpuct'] * self.Ps[state][action] * sqrt(self.Ns[state] + EPS)  # Q = 0 ?

                if u > best:
                    best = u
                    best_act = action

        action = best_act
        next_s, next_player = self.game.getNextState(board, 1, action)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + v) / (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1

        else:
            self.Qsa[(state, action)] = v
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return -v        
