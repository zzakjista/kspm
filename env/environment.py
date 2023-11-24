from dataset.preprocess import Preprocesser

class Env:

    def __init__(self, dataset, args):
        self.ppc = Preprocesser(args)
        self.stock_code = args.stock_code
        self.mode = args.template
        self.dataset = dataset
        
        self.chart_data = None
        self.state_data = None
        
        self.state = None
        self.chart = None

        self.x_window_size = args.x_window_size
        self.current_idx = self.x_window_size # chart data의 인덱스
        self.state_idx = 0 # state data의 인덱스
        
    def reset(self):
        self.current_idx = self.x_window_size - 1
        self.state_idx = 0 
        self.done = False

    def current_price(self):
        return self.chart['종가'][self.current_idx] 
    
    def current_date(self):
        return self.chart.index[self.current_idx]
    
    def next_price(self):
        return self.chart['종가'][self.current_idx + 1]

    def get_state(self):
        return self.state[self.state_idx]
    
    def get_next_state(self):
        return self.state[self.state_idx + 1]

    def get_last_state(self):
        return self.state[-1]

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