from Coach import Coach
from Game import Game
from NetworkWrapper import NetworkWrapper

args = {
    'iters': 1000,
    'episodes': 100,
    'tempThreshold': 15,
    'updateThreshold': '0.6',
    'maxlenOfQueue': 200000,
    'numSims': 25,
    'n': 40,   
    'cpuct': 1,
    
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
}

def main():
    game = Game()
    network = NetworkWrapper(game)

    try:
        network.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    except:
        pass

    coach = Coach(game, network, args)

    if args['load_model']:
        coach.loadTrainExamples()

    coach.learn()

if __name__ == '__main__':
    main()