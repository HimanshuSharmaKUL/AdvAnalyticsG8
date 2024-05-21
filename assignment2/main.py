# main.py

import argparse
from Trainer import main as train_model

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=32)
    parser.add_argument('--lr',            type=float,  default=1e-4,     help="initial learning rate")
    parser.add_argument('--num_epoch',     type=int, default=50,     help="number of total epoch")
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--DR',            type=str, default='images/',  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--per_save',      type=int, default=5,      help="Save checkpoint every seted epoch")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")  
    args = parser.parse_args()

    train_model(args)

if __name__ == '__main__':
    main()

