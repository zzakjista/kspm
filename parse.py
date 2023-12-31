import argparse
from templates import set_template

parser = argparse.ArgumentParser(description='Reinforcement Learning with Stock Trading')

parser.add_argument('--template', type=str, default='train', choices = ['train','test']) # train, test

# path #
parser.add_argument('--rawdata_path', type=str, default='kospi200_10Y_price_investor.pickle')
parser.add_argument('--preprocessed_data_path', type=str, default='preprocessed_data.pickle')

# preprocessing #
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--test_ratio', type=float, default=0.1)

parser.add_argument('--scaler', type=str, default='standard_scaler', choices=['standard_scaler','minmax_scaler','robust_scaler'])
parser.add_argument('--x_window_size', type=int, default=20, choices=[5,10,20])

## reinforce Learning ##
# env # 
parser.add_argument('--stock_code', type=str, default='005930')
# agent #
parser.add_argument('--initial_balance', type=int, default=5000000000)
parser.add_argument('--min_trading_unit', type=int, default=1)
parser.add_argument('--max_trading_unit', type=int, default=10)
parser.add_argument('--model_name', type=str, default='VALSTM', choices=['VALSTM','CNNLSTM'])

# policy #
parser.add_argument('--action_kind', type=int, default=3) 
# critic #
parser.add_argument('--value_kind', type=int, default=1) 
# policy & critic parameter share # 
parser.add_argument('--in_channel', type=int, default=19)
parser.add_argument('--out_channel', type=int, default=64)
parser.add_argument('--input_size', type=int, default=19)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--temperature', type=int, default=3)
parser.add_argument('--temperature_decay', type=float, default=0.99)

# train # 
parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_episode', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.8) # discount rate

# load_version # 
parser.add_argument('--pretrained', type=bool, default=False) # 
parser.add_argument('--version', type=str, default='v1.1') # v1.0

# Replay Memory # 
parser.add_argument('--capacity', type=int, default=100)

# seed #
parser.add_argument('--seed', type=int, default=21)

# args = parser.parse_args()  # on the terminal
args = parser.parse_args(args=[]) # on the jupyter notebook
set_template(args)