from dataset.preprocess import Preprocesser

class Env:

    def __init__(self, dataset, args):
        # preprocessed data를 받아서 policy에 들어가는 state의 순서에 맞게 차트를 가져올 수 있어야한다
        # KOSPI 200을 받은 후 개별 종목으로 환경 구성 
        self.ppc = Preprocesser(args)
        self.stock_code = args.stock_code
        self.mode = args.template
        self.dataset = dataset
        
        self.chart_data = None
        self.state_data = None
        self.config_environment(self.stock_code) # 지정된 window_size에서 state를 구성
        
        self.state = None
        self.chart = None

        self.x_window_size = args.x_window_size
        self.current_idx = self.x_window_size # chart data의 인덱스
        self.state_idx = 0 # state data의 인덱스
        
    def reset(self):
        self.current_idx = self.x_window_size
        self.state_idx = 0 
        self.done = False
        # self.total_profit = 0

    def current_price(self):
        return self.chart['시가'][self.current_idx] # 현재는 다음날 시가에 거래되게 되어있음

    def get_state(self):
        return self.state[self.state_idx]
    
    def get_next_state(self):
        return self.state[self.state_idx + 1]

    def config_environment(self, stock_code): # 종목코드를 받아서 환경을 구성
        data = self.dataset[stock_code]

        chart_columns = ['시가', '고가', '저가', '종가']
        chart = data[chart_columns]
        if self.mode == 'train':
            train, val, test = self.ppc.split_data(chart)
            self.chart_data = {'train': train, 'val': val, 'test': test}
        elif self.mode == 'test':
            self.chart_data = {'test': chart}
        self.state_data = data.drop(columns=chart_columns)
        self.state_data = self.ppc.make_preprocessed_data(stock_code, self.state_data)