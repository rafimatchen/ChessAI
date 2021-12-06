import chess
import numpy as np
import copy

class Game():
    def __init__(self):
        pass
    
    def getInitBoard(self):
        return chess.Board()
    
    def getBoardSize(self):
        return (8, 8)
    
    def getActionSize(self):
        return 8 * 8 * (8 * 7 + 8 + 9)

    def getNextState(self, board, player, action):
        board.push(list(board.legal_moves)[action])
        return board, player

    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        b = copy.deepcopy(board)
        moves = list(b.legal_moves)
        if len(moves) == 0:
            return np.array(valids)
        for i in range(len(moves)):
            valids[i] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        outcome = board.outcome()
        if outcome is None:
            return 0
        if outcome.winner == chess.WHITE:
            return 1
        return -1


    def getBoard(self):
        return self.board

    def stringRepresentation(self, board):
        return board.fen()