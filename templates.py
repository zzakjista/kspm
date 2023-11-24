
def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train'):
        args.rawdata_path = 'kospi200_10Y_price_investor.pickle'
        args.preprocessed_data_path = 'preprocessed_data.pickle'
        args.train_ratio = 0.7
        args.val_ratio = 0.2
        args.test_ratio = 0.1
        args.scaler = 'standard_scaler'
        args.x_window_size = 10 # 학습 시 사용할 x_window_size
        # agent #
        args.initial_balance = 500000000
        args.min_trading_unit = 1
        # args.max_trading_unit = 100
        args.device = 'cpu'
        args.model_name = 'CNNLSTM'

        # LSTM #
        args.input_size = 19
        args.hidden_size = 64
        args.num_layers = 3
        args.dropout = 0.2
        
        args.gamma = 0.8
        args.lr = 0.0007
        args.temperature = 1
        args.temperature_decay = 0.98
        args.max_episode = 1
        args.batch_size = 16
        args.action_kind = 3
        args.capacity = 1000
        args.seed = 10000000

        # 학습 이어갈 때 유의 사항 #
        # 1. 기존 하이퍼 파라미터 변경하지 않기
        # 2. 하위 경로를 잘 체크하기
        # 3. pretrained 안쓴다면 꼭 False 해주기
        args.pretrained = True 
        args.version = 'v2.27' # 실험 버전
        args.pretrained_version = 'pretrained.5' # 이어서 학습할 버전이 있다면 하위 경로 기재 ex) pretrained.0, 아니라면 ''


        
        