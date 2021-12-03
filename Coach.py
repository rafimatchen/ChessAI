import os
import sys
import pickle
import random
import numpy as np 
import PlayWrapper
import MCTS
import NetworkWrapper

class Coach():

    def __init__(self, game, network, args):
        self.game = game
        self.network = network
        self.args = args
        self.pnet = NetworkWrapper.NetworkWrapper(self.game)
        self.mcts = MCTS.MCTS(self.game, self.network, self.args)
        self.history = []

    def episode(self):
        examples = []
        board = self.game.getInitBoard()
        self.current = 1
        step = 0

        while True:
            step += 1
            #board = self.game.getBoard()
            temp = int(step < self.args['tempThreshold'])

            pi = self.mcts.getActionProbability(board, temp=temp)
            action = np.random.choice(len(pi), p=pi)
            board, self.current = self.game.getNextState(board, self.current, action)

            r = self.game.getGameEnded(board, self.current)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.current))) for x in examples]
    
    def learn(self):
        for i in range(1, self.args['iters'] + 1):
            examples = []

            for j in range(self.args['episodes']):
                self.mcts = MCTS.MCTS(self.game, self.network, self.args)
                examples += self.episode()

            self.history.append(examples)
            if len(self.history) > self.args['historyIters']:
                del self.history[-1]

            self.saveExamples(i - 1)

            trainExamples = []
            for e in self.history:
                trainExamples.extend(e)
            np.shuffle(trainExamples)

            self.network.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.network.train(trainExamples)
            nmcts = MCTS(self.game, self.network, self.args)

            playwrapper = PlayWrapper(lambda x: np.argmax(pmcts.getActionProbability(x, temp=0)),
                                      lambda x: np.argmax(nmcts.getActionProbability(x, temp=0)), 
                                      self.game)
            pWins, nWins, draws = playwrapper.playMany(self.args['n'])
            if pWins + nWins == 0 or float(nWins) / (pWins + nWins) < self.args['updateThreshold']:
                self.network.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            else:
                self.network.save_checkpoint(folder=self.args['checkpoint'], filename=self.getCheckpointFile(i))
                self.network.save_checkpoint(folder=self.args['checkpoint'], filename='best.pth.tar')

    def getCheckpointFile(self, i):
        return 'checkpoint_' + str(i) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            pickle.Pickler(f).dump(self.history)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            with open(examplesFile, "rb") as f:
                self.history = pickle.Unpickler(f).load()

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True