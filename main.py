import os 
import argparse
from dataset.dataset import get_loader
from solver import Solver

def main(config):
    test_loader = get_loader(test_mode=config.test_mode, sal_mode=config.sal_mode)
    if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
    test = Solver(test_loader, config)
    test.test(test_mode=config.test_mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--cuda', type=bool, default=True)

    # Testing settings
    parser.add_argument('--model', type=str, default='pretrained/dfi.pth')
    parser.add_argument('--test_fold', type=str, default='demo/predictions')
    parser.add_argument('--test_mode', type=int, default=3) # choose task
    parser.add_argument('--sal_mode', type=str, default='e') # choose dataset, details in 'dataset/dataset.py'

    config = parser.parse_args()
    main(config)
