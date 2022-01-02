from train import Train
from threading import main_thread
from option import args
from model import SRCNN
from dataset import *


if __name__ == "__main__":
    t=Train(args)
    t.train()