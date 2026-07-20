import os
import torch
from torch.backends import cudnn
from config import get_config
from solver import Solver

def main(config):

    cudnn.benchmark = True
    
    solver = Solver(config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'test_one':
        solver.test_one()


if __name__ == '__main__':
    config = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    config.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    main(config)

#
