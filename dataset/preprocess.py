import talib
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from path_control import path_master

class Preprocesser:

    def __init__(self,args):
        self.mode = args.template

        self.co_short_period = 5 
        self.co_long_period = 20
        self.vr_period = 20

        self.train_ratio = args.train_ratio
        self.val_ratio = args.val_ratio
        self.test_ratio = args.test_ratio
        self.x_window_size = args.x_window_size # train set window size
        self.scaler = args.scaler
        
        self.raw_data_path = args.rawdata_path
        self.preprocessed_data_path = args.preprocessed_data_path
        self.path_master = path_master(args)

    def make_dataset(self):
        ppc_folder_path = self.path_master.get_preprocessed_folder_path()
        ppc_file_path = ppc_folder_path.joinpath(self.preprocessed_data_path)

        if ppc_file_path.exists():
            print('load preprocessed data')
            with open(ppc_file_path, 'rb') as f:
                data = pd.read_pickle(f)
        else:
            print('preprocess raw data')
            raw_folder_path = self.path_master.get_rawdata_folder_path()
            raw_file_path = raw_folder_path.joinpath(self.raw_data_path)
            with open(raw_file_path, 'rb') as f:
                data = pd.read_pickle(f)
            data = self.preprocess(data)
            ppc_folder_path = self.path_master.get_preprocessed_folder_path()
            ppc_file_path = ppc_folder_path.joinpath(self.preprocessed_data_path)
            with open(ppc_file_path, 'wb') as f:
                pickle.dump(data, f)
        # dataset = self.make_each_stock_dataset(data) # 강화학습용 데이터셋 -> 환경에서 configuration
        # dataset = self.make_all_stock_dataset(dataset) # 단일 모델용 데이터셋 {train:{X,y},val:{X,y},test:{X,y}}
        return data
        
    def make_each_stock_dataset(self, data): # 각 주식 데이터를 하나의 데이터셋으로 만듦
        dataset = {}
        keys = list(data.keys())
        for key in keys:
            key_data = self.make_preprocessed_data(data[key])
            dataset[key] = key_data
        return dataset
    
    def make_preprocessed_data(self, stock_code, df): # 버전에 따른 전처리 :  X,y 생성 -> train, test 분리 -> label 생성 -> scaling 
        key_data = {}
        X = df.values.reshape(df.shape[0],-1)
        if self.mode == 'test': # test set은 전체 데이터를 사용
            X, _ , _ = self.scaling_data(stock_code, X, None, None)
            X = self.make_sequence_x(X)
            key_data['test'] = X

        elif self.mode == 'train': # train set은 train, val, test로 분리
            X = self.split_data(X) 
            try:
                X_train, X_val, X_test = X[0], X[1], X[2]
                # y_train, y_val, y_test = self.make_label(y[0]), self.make_label(y[1]), self.make_label(y[2])

                X_train, X_val, X_test = self.scaling_data(stock_code, X_train, X_val, X_test)
                # y_train, y_val, y_test = self.scaling_data(y_train, y_val, y_test)

                X_train, X_val, X_test = self.make_sequence_x(X_train), self.make_sequence_x(X_val), self.make_sequence_x(X_test)

                key_data['train'] = X_train
                key_data['val'] = X_val
                key_data['test'] = X_test    

                # key_data['train'] = y_train
                # key_data['val'] = y_val
                # key_data['test'] = y_test
            except:
                pass
        return key_data

    def split_data(self, data):
        train_size = int(len(data) * self.train_ratio)
        val_size = int(len(data) * self.val_ratio)
        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]
        test_data = data[train_size+val_size:]
        return train_data, val_data, test_data

    
    def scaling_data(self, stock_code, train, val, test):
        scaler_path = self.path_master.get_scaler_root_path().joinpath(f'{self.scaler}.pkl')
        scaler_set = self.call_scaler_set(scaler_path)
        scaler, reuse = self.call_scaler(stock_code, scaler_set)

        if not reuse:
            scaler.fit(train)
            scaler_set[stock_code] = scaler 
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_set, f)
        try:
            train = scaler.transform(train)
        except:
            train = None
        try:
            val = scaler.transform(val)
        except:
            val = None
        try:
            test = scaler.transform(test)
        except:
            test = None
        return train, val, test
    
    def call_scaler(self, stock_code, scaler_set):
        if stock_code in scaler_set.keys():
            scaler = scaler_set[stock_code]
            reuse = True
        else:
            if self.scaler == 'standard_scaler':
                scaler = StandardScaler()
            elif self.scaler == 'minmax_scaler':
                scaler = MinMaxScaler()
            elif self.scaler == 'robust_scaler':
                scaler = RobustScaler()
            reuse = False
        return scaler, reuse

    def call_scaler_set(self, scaler_path):
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler_set = pickle.load(f)
        else:
            scaler_set = {}
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_set, f)
        return scaler_set

    def make_sequence_x(self, X): # x_window_size를 받아 X의 state sequence를 시퀀스를 생성
        seq = []
        for idx in range(0,len(X)-self.x_window_size+1): # 강화학습이라면 y_window_size를 고려하지 않아도됨 -self.y_window_size
            x = X[idx : idx + self.x_window_size,:]
            seq.append(x)
        return np.array(seq)

    def preprocess(self, data): # 필수 전처리 : 거래정지 -> 지표생성 -> faeture, 결측치 제거 
        keys = list(data.keys())
        for key in keys:
            data[key] = self.drop_zero_volume(data[key])
            data[key] = self.generate_volume_indicators(data[key])
            data[key] = self.generate_price_indicators(data[key])
            data[key] = self.transform_price(data[key])
            data[key]['거래량'] = np.log(data[key]['거래량'])
            # data[key] = data[key].drop(columns=['수정전거래량','수정전종가','전체'])
            data[key] = data[key].dropna()

        return data

    # Generate Indicator by using Price #
    # 가격대에 따라 가격의 차이는 상대적이기 때문에 종가 대비 가격 비율을 사용 #
    def transform_price(self, df):
        df['종대시'] = (df['시가'] - df['종가']) / df['종가']
        df['종대고'] = (df['고가'] - df['종가']) / df['종가']
        df['종대저'] = (df['저가'] - df['종가']) / df['종가']
        return df

    # Generate Indicator by using Volume #

    def generate_volume_indicators(self, df):
        df['OBV'] = self.OnBalanceVolume(df)
        df['PVT'] = self.PriceVolumeTrend(df)
        df['AD'] = self.Accumulation_Distribution(df)
        df['CO'] = self.chaikin_oscillator(df, self.co_short_period, self.co_long_period)
        df['VR'] = self.VolumeRatio(df, self.vr_period)
        return df

    def OnBalanceVolume(self, df):
        OBV = []
        OBV.append(df['거래량'][0])
        for i in range(1, len(df)):
            if df['종가'][i] > df['종가'][i-1]:
                OBV.append(OBV[-1] + df['거래량'][i])
            elif df['종가'][i] < df['종가'][i-1]:
                OBV.append(OBV[-1] - df['거래량'][i])
            else:
                OBV.append(OBV[-1])
        return OBV

    def PriceVolumeTrend(self,df):
        PVT = []
        PVT.append(df['거래량'][0])
        for i in range(1, len(df)):
            PVT.append(((df['종가'][i] - df['종가'][i-1]) / df['종가'][i-1]) * df['거래량'][i] + PVT[-1])
        return PVT

    def Accumulation_Distribution(self,df):
        AD = []
        AD.append(df['거래량'][0])
        for i in range(1, len(df)):
            if df['고가'][i] != df['저가'][i]:
                AD.append(AD[-1] + ((df['종가'][i] - df['저가'][i]) - (df['고가'][i] - df['종가'][i])) / (df['고가'][i] - df['저가'][i]) * df['거래량'][i])
            else:
                AD.append(AD[-1])
        return AD

    def chaikin_oscillator(self, df, short_period, long_period):
        if 'AD' not in df.columns:
            df['AD'] = self.Accumulation_Distribution(df)
        AD_short = df['AD'].rolling(short_period).mean()
        AD_long = df['AD'].rolling(long_period).mean()
        CO = AD_short - AD_long
        return CO

    def VolumeRatio(self, df, period):
        up = np.where(df['등락률']>0, df['거래량'], 0)
        down = np.where(df['등락률']<0, df['거래량'], 0)
        same = np.where(df['등락률']==0, df['거래량'], 0)
        up = pd.Series(up).rolling(period).sum()
        down = pd.Series(down).rolling(period).sum()
        same = pd.Series(same).rolling(period).sum()
        VR = (up + same/2) / (down + same/2) * 100
        return list(VR)

    def generate_price_indicators(self, df): # window size에 따라 최적의 지표가 있지않을까?
        df['RSI'] = self.RSI(df)
        df['STOCH'] = self.STOCH(df)
        df['MACD'] = self.MACD(df)
        df['BB_P_B'] = self.BBANDS(df)
        df['AROON_OSC'] = self.AROONOSC(df)
        return df

    def RSI(self, df):
        RSI = talib.RSI(df['종가'])
        return RSI
    
    def STOCH(self, df):
        slowk, slowd = talib.STOCH(df['고가'], df['저가'], df['종가'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        return slowk
    
    def MACD(self, df):
        macd, macdsignal, macdhist = talib.MACD(df['종가'], fastperiod = 12, slowperiod = 26, signalperiod = 9)
        return macdhist
    
    def BBANDS(self, df):
        up_band, mid_band, lw_band = talib.BBANDS(df['종가'], timeperiod = 20, nbdevup = 2, nbdevdn = 2, matype = 0)
        BB_P_B = (df['종가'] - lw_band) / (up_band - lw_band)
        return BB_P_B

    def AROONOSC(self, df):
        AROON_OSC = talib.AROONOSC(df['고가'], df['저가'], timeperiod=14)
        return AROON_OSC

    # 거래정지된 데이터 제거 # 
    def drop_zero_volume(self, df):
        df = df[df['거래량']!=0]
        return df
