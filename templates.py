
def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('www'):
        args.rawdata_path = 'kospi200_10Y_price_investor.pickle'
        args.preprocessed_data_path = 'preprocessed_data.pickle'
        args.train_ratio = 0.7
        args.val_ratio = 0.2
        args.test_ratio = 0.1
        args.scaler = 'standard_scaler'
        args.x_window_size = 20 # 학습 시 사용할 x_window_size
        args.y_window_size = 5 # 학습 시 사용할 y_window_size

        # agent #
        args.initial_balance = 1000000

        args.device = 'cpu'

        # LSTM #
        args.model = 'LSTM'
        args.num_layers = 2
        args.dropout = 0.2
        

        # training
        args.eps_start = 0.9
        args.eps_end = 0.1
        args.eps_decay = 200
        
        args.gamma = 0.99
        args.lr = 0.005
        args.max_episode = 10
        args.batch_size = 32
        args.action_kind = 3
        args.capacity = 1000
        
