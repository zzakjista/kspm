# coding: utf-8
# Library
import pandas as pd 
import numpy as np 
import os
import inspect
from scipy.stats import zscore

# 데이터 프레임의 이름을 저장하기 위한 메소드 
def dataframe_name(df):
    frame = inspect.currentframe().f_back
    variable_name = [k for k, v in frame.f_globals.items() if v is df][0]
    return variable_name

# Profitabilty 계산 메서드
def profitability(temp_df):
    
    df_profitability = pd.DataFrame()
    df_profitability['GPOA'] = (temp_df['Gross Profit'] /temp_df['Total Asset'])
    df_profitability['ROE'] = (temp_df['Net Income'] / temp_df['Total Equity'])
    df_profitability['ROA'] = ((temp_df['Net Income'] / temp_df['Total Asset']))
    df_profitability['CFOA'] = (temp_df['Operating Cashflow'] / temp_df['Total Asset'])
    df_profitability['ACC'] = (temp_df['Net Income'] - temp_df['Operating Cashflow'])/temp_df['Total Asset']
    df_profitability = df_profitability.fillna(0.01)
    
    return df_profitability

#  Safety 계산 메서드
def safety(temp_df):
    df_safety = pd.DataFrame()

    A = temp_df['Working Asset']/temp_df['Total Asset']
    B = temp_df['Retained Earnings'] / temp_df['Total Asset']
    C = temp_df['EBIT'] / temp_df['Total Asset']
    D = temp_df['Total Asset'] / temp_df['Total Liabilities']
    E = temp_df['Revenue'] / temp_df['Total Asset']
    temp_df['altman']= 1.2 * (A) + 1.4 * (B) + 3.3 * C + 0.6*D + 1.0*E 

    df_safety['Leverage'] = (temp_df['Total Liabilities'] / temp_df['Total Asset'])
    df_safety['Gross Profit/Total Assets Ratio'] = (temp_df['Gross Profit']/ temp_df['Total Asset'])
    df_safety['Operating Cash Flow/Total Assets Ratio'] = (temp_df['Operating Cashflow']/ temp_df['Total Asset'])
    df_safety['altman'] =(temp_df['altman'])
    df_safety = df_safety.fillna(0.01)
    
    return df_safety

# growth 계산 메서드
def growth(temp_df):
    df_profitability = profitability(temp_df).iloc[:,:-1]
    df_growth = df_profitability.pct_change()[::-1] # 역순으로 pct_change를 해서 최근 Growth 변화량 계산. 
    df_growth = df_growth.applymap(lambda x: round(x,3))
    df_growth = df_growth.dropna()
    df_growth = df_growth.fillna(0.01)

    return df_growth

# 퀄리티 계산 메서드
def Quality(sector_profitability, sector_safety, sector_growth):
    stock_sector = pd.DataFrame()
    temp_profitability = zscore(sector_profitability.iloc[:,:-1])
    temp_safety = zscore(sector_safety.iloc[:,:-1])
    temp_growth = zscore(sector_growth.iloc[:,:-1])

    stock_sector['Profitability'] = zscore(temp_profitability.sum(axis =1))
    stock_sector['Safety'] = zscore(temp_safety.sum(axis =1))
    stock_sector['Growth'] = zscore(temp_growth.sum(axis =1))

    stock_sector['Quality'] = zscore(stock_sector.sum(axis=1))
    stock_sector['invest_opinion'] = 'long' 
    stock_sector.loc[stock_sector['Quality'] <= 0, 'invest_opinion'] = 'short'

    return stock_sector


map_columns = {'자본총계':'Total Asset', '유동자산':'Working Asset','자산총계':'Total Equity',
'부채총계':'Total Liabilities','이익잉여금(결손금)':'Retained Earnings',
'이익잉여금':'Retained Earnings','6.이익잉여금':'Retained Earnings',
'수익(매출액)':'Revenue','매출액':'Revenue', '매출':'Revenue','매출액(영업수익)':'Revenue', 
'매출총이익':'Gross Profit','매출총이익(손실)':'Gross Profit', 
'영업이익':'Operating Income','영업이익(손실)':'Operating Income',
'당기순이익(손실)':'Net Income','당기순이익':'Net Income','당기순손익':'Net Income',
'법인세차감전순이익':'EBIT','법인세차감전순이익(손실)':'EBIT','법인세비용차감전계속영업순이익':'EBIT',
'법인세비용차감전계속영업순이익(손실)':'EBIT','법인세비용차감전순이익(손실)':'EBIT',
'법인세비용차감전이익':'EBIT','법인세비용차감전이익(손실)':'EBIT',
'법인세비용차감전순이익':'EBIT','법인세비용차감전손실':'EBIT',
'법인세비용차감전순손익':'EBIT','법인세비용차감전순손익(손실)':'EBIT',
'영업활동 현금흐름':'Operating Cashflow','영업활동현금흐름':'Operating Cashflow','영업활동순현금흐름':'Operating Cashflow',
'영업활동으로인한현금흐름':'Operating Cashflow', '영업활동으로 인한 현금흐름':'Operating Cashflow',
'영업활동으로 인한 순현금흐름':'Operating Cashflow','영업활동으로부터의현금흐름':'Operating Cashflow'}

# 폴더 경로 설정
folder_path = '../data/stock_balance_data_year/'
# 폴더 내의 파일 목록 가져오기
file_list = os.listdir(folder_path)
# 각 파일을 불러와서 데이터프레임으로 저장
stock_dict = {} 
quality_df_list = {}

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    # 파일을 데이터프레임으로 읽어서 딕셔너리에 저장
    df = pd.read_csv(file_path,encoding='euc-kr',index_col=0)
    # 파일명을 키로 사용하여 데이터프레임 저장
    df = df.rename(columns = map_columns)
    # 결측값 부동소수점으로 대체
    df = df.fillna(0.01) 
    stock_dict[file_name[:-12]] = df
    
    stock_profitability = profitability(df)
    stock_safety = safety(df)
    stock_growth = growth(df)
    stock_quality = Quality(stock_profitability, stock_safety, stock_growth).sort_index(ascending=True)
    quality_df_list[file_name[:-12]] = stock_quality    
    
    # csv 파일로 저장
    quality_df_list[file_name[:-12]]['stock_name'] = file_name[:-12]
    quality_df_list[file_name[:-12]].to_csv(f'../data/stock_quality_data/{file_name}_quality.csv',encoding='euc-kr')