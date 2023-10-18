import argparse
from templates import set_template

parser = argparse.ArgumentParser(description='RecPlay')

parser.add_argument('--template', type=str, default='train', choices = ['train','test'])

# data #
parser.add_argument('--rawdata_path', type=str, default=None)
parser.add_argument('--preprocessed_data_path', type=str, default=None)

# preprocessing #
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--test_ratio', type=float, default=0.1)

parser.add_argument('--scaler', type=str, default='standard_scaler', choices=['standard_scaler','minmax_scaler','robust_scaler'])
parser.add_argument('--x_window_size', type=int, default=None)
parser.add_argument('--y_window_size', type=int, default=None)




## reinforce Learning ##

# env # 
parser.add_argument('--stock_code', type=str, default='005930')

# agent #
parser.add_argument('--initial_balance', type=int, default=1000000)

# policy #
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)

# train #
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'])
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'])


parser.add_argument('--action_kind', type=int, default='3') # buy, sell , hold
parser.add_argument('--max_episode', type=int, default='10')
parser.add_argument('--eps_start', type=float, default='0.9')
parser.add_argument('--eps_end', type=float, default='0.1')
parser.add_argument('--eps_decay', type=float, default='0.99')
parser.add_argument('--gamma', type=float, default='0.99') # discount rate

# Replay Memory # 
parser.add_argument('--capacity', type=int, default='1000')

# args = parser.parse_args()  # on the terminal
args = parser.parse_args(args=[]) # on the jupyter notebook
set_template(args)