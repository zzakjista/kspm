from pykrx import stock
from pykrx import bond
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import time
import random


def scrap_data(ISCD, start_date, end_date):
    # 투자자별 거래대금 가져오기 
    volume_value = stock.get_market_trading_value_by_date(start_date, end_date, ISCD)
    volume_value.drop(['전체'], axis=1, inplace=True)
    # 기본정보 가져오기
    price_adj = stock.get_market_ohlcv(start_date, end_date, ISCD,adjusted=True) #종목별/기간별 시가, 고가, 저가, 종가, 거래량
    price_no_adj = stock.get_market_ohlcv(start_date, end_date, ISCD,adjusted=False) #종목별/기간별 시가, 고가, 저가, 종가, 거래량, 거래대금
    price_adj['거래량'] = (price_no_adj['거래대금'] / ((price_adj['시가']+price_adj['고가']+price_adj['저가']+price_adj['종가']) / 4 )).astype('int') # 근사치
    data = price_adj.merge(volume_value, how='left', left_index=True, right_index=True)
    return ISCD, data

def save_data(ISCD, data, path='data'):
    path = Path(path+'.pickle')
    tmp = {ISCD : data}
    if not path.is_file():
        with open(path, 'wb') as f:
            pickle.dump(tmp, f)
    else:
        with open(path, 'rb') as f:
            tmp = pickle.load(f)
        tmp[ISCD] = data
        with open(path, 'wb') as f:
            pickle.dump(tmp, f)
    print(ISCD, '저장완료')




