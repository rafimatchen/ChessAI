class PlayWrapper():
    def __init__(self, p1, p2, game):
        self.p1 = p1
        self.p2 = p2
        self.game = game

    def playOnce(self):
        players = [self.p1, None, self.p2]
        current = 1
        board = self.game.getInitBoard()
        i = 0
        while not self.game.getGameEnded(board, current):
            i += 1
            action = players[current + 1](board)
            valids = self.game.getValidMoves(board, 1)

            board, current = self.game.getNextState(board, current, action)

        return current * self.game.getGameEnded(board, current)

    def playMany(self, n):
        n = int(n / 2)
        oneWins = 0
        twoWins = 0
        draws = 0

        for i in range(n):
            result = self.playOnce()
            if result == 1:
                oneWins += 1
            if result == -1:
                twoWins += 1
            else:
                draws += 1
        
        self.p1, self.p2 = self.p2, self.p1

        for i in range(n):
            result = self.playOnce()
            if result == -1:
                oneWins += 1
            if result == 1:
                twoWins += 1
            else:
                draws += 1

        return oneWins, twoWins, draws