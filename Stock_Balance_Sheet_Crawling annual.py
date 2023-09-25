## Library Import
import pandas as pd 
import numpy as np 
import requests
import inspect
import re
import OpenDartReader
import dart_fss as dart
from pykrx import stock
from pykrx import bond
from scipy.stats import zscore

# 객체 생성
api_key = "af983cf5135fd27b1b42edb107d1854207fd26c6"
# DATR에서 API키 설정 후 요청 인증
dart.set_api_key(api_key=api_key)
# DART 기업 목록
corp_list = dart.get_corp_list()

## 종목코드, 종목명 파일 업로드
kospi200 = pd.read_csv('../data/kospi200_0826.csv',encoding='euc-kr')
kospi200['종목코드'] = kospi200['종목코드'].astype('str')

def add_zeros(code):
    if len(code) == 2:
        return '0000' + code
    elif len(code) == 3:
        return '000' + code
    elif len(code) == 4:
        return '00' + code
    elif len(code) == 5:
        return '0' + code
    else:
        return code

kospi200['종목코드_process'] = kospi200['종목코드'].apply(add_zeros)
kospi200_code_name = kospi200[['종목코드_process','종목명']]
kospi200_code_name = kospi200_code_name.rename(columns = {'종목코드_process':'종목코드'})

## Dart 데이터 전처리 메소드
# bs : 재무상태표, Is : 손익계산서(포괄손익계산서 대신 사용), cf : 현금흐름표
def preprocess_dart1(bs,cis,Is,cf):
    # 행렬 변환
    if bs is not None:
        bs = bs.transpose() 
        # header지정 및 불필요한 칼럼 제거
        new_header = bs.iloc[0]
        bs = bs[1:]
        bs.columns = new_header
        bs.drop(bs.index[0:2],inplace= True)

    if cis is not None:
        cis = cis.transpose()
        new_header = cis.iloc[0]
        cis = cis[1:]
        cis.columns = new_header
        cis.drop(cis.index[0:2],inplace= True)

    if Is is not None:
        Is = Is.transpose()                
        new_header = Is.iloc[0]
        Is = Is[1:]
        Is.columns = new_header
        Is.drop(Is.index[0:2],inplace= True)    
    
    if cf is not None:
        cf = cf.transpose()                
        new_header = cf.iloc[0]
        cf = cf[1:]
        cf.columns = new_header
        cf.drop(cf.index[0:2],inplace= True)

    return bs, cis, Is, cf 

def preprocess_dart2(bs,cis,Is,cf):
    # 계산을 위한 index reset, 기존 연도 index는 내림차순
    
    if bs is not None:
        new_index = [idx[0].split('\t')[0][:4] for idx in bs.index]
        bs.index = new_index
    
    if cis is not None:
        new_index = [idx[0].split('\t')[0][:4] for idx in cis.index]
        cis.index = new_index

    if Is is not None:    
        new_index = [idx[0].split('\t')[0][:4] for idx in Is.index]
        Is.index = new_index

    if cf is not None:
        new_index = [idx[0].split('\t')[0][:4] for idx in cf.index]
        cf.index = new_index

    # 계산을 위해 balance sheet, income statement, cashflow merge
    merged_df = pd.concat([bs,cis,Is,cf], axis=1, join = 'inner')
    return merged_df


## 데이터 프레임의 이름을 저장하기 위한 메소드 
def dataframe_name(df):
    frame = inspect.currentframe().f_back
    variable_name = [k for k, v in frame.f_globals.items() if v is df][0]
    return variable_name

## 인덱스의 이름을 해당 기업의 이름으로 바꾸기 위한 메소드 
def change_index(df,df2):
    name = dataframe_name(df2)
    df = df.rename(index = {0: name})
    return df 

## 정규표현식 사용하여 유효하지 않은 문자 제거
def clean_company_name(company_name):
    clean_name = re.sub(r'[\\/:"*?<>|]+', '', company_name)
    return clean_name

## 재무제표 데이터 수집
# 빈 오류 데이터프레임 생성
kospi200_code_name = kospi200_code_name[:2]

error_df = pd.DataFrame(columns=['Ticker', 'Company Name', 'Error Message'])
for ticker, company_name in zip(kospi200_code_name['종목코드'], kospi200_code_name['종목명']):
    try:
        # Clean the company name
        cleaned_company_name = clean_company_name(company_name)

        # bs: 재무상태표, Is: 손익계산서(포괄손익계산서 대신 사용), cf: 현금흐름표
#        df = dart.fs.extract(ticker, '20160101', end_de='20231231', fs_tp=('bs','cis','is','cf'), separate=False, report_tp='annual', lang='ko', separator=True, dataset='xbrl', cumulative=False, progressbar=True, skip_error=True,last_report_only=True)
        df = dart.fs.extract(ticker, '20160101', end_de='20231231', fs_tp=('bs','cis','is','cf'), separate=False, report_tp='quarter', lang='ko', separator=True, dataset='xbrl', cumulative=False, progressbar=True, skip_error=True, last_report_only=True)
        df.to_dict()

        bs = df.show('bs', show_class=False, show_depth=0, show_concept=False)
        cis = df.show('cis', show_class=False, show_depth=0, show_concept=False)
        Is = df.show('is', show_class=False, show_depth=0, show_concept=False)
        cf = df.show('cf', show_class=False, show_depth=0, show_concept=False)
        bs, cis, Is, cf = preprocess_dart1(bs, cis, Is, cf)

        # Quality 계산을 위해 필요한 column
        stock_balance = preprocess_dart2(bs, cis, Is, cf)

        # 파일로 저장
        stock_balance.to_csv(f'./stock_balance_data_quarter/{cleaned_company_name}_balance_quarter.csv', encoding='euc-kr')

    except Exception as e:
        # 오류 발생 시 데이터프레임에 오류 정보 추가
        error_df = error_df.append({'Ticker': ticker, 'Company Name': company_name, 'Error Message': str(e)}, ignore_index=True)
        print(f"Error for ticker {ticker}: {str(e)}")
        pass


    1+3