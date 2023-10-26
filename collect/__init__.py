from .collection import *
import pandas as pd

def collect_stock():
    dataset = {}
    ISCD = input('종목코드(6자리)를 입력하세요 : ')
    start_date = input('시작일을 입력하세요(YYYYMMDD) : ')
    end_date = input('종료일을 입력하세요(YYYYMMDD) : ')
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError('datetime error')
    file_name = input('저장할 파일명을 입력하세요(default=data) : ')
    ISCD, data = scrap_data(ISCD, start_date, end_date)
    dataset[ISCD] = data    
    return dataset
    if input('저장하시겠습니까?: y/n' ) == 'y':
        save_data(iscd, data, file_name)
