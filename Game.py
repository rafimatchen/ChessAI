import chess
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
        board.push(action)
        return board

    def getValidMoves(self, board, player):
        return board.legal_moves

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