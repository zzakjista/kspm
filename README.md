# REINFORCE TRADER IN KOSPI 200

## 개요
- 강화학습 에이전트가 수익률을 극대화하는 거래를 할 수 있도록 학습하고 KOSPI 200 종목에서 실전 매매를 진행

## 배경 
- 딥러닝을 이용하여 주식 가격을 예측하는 프로젝트는 일반적으로 시도되었으나, 단순 예측에 기반하기 때문에 투자 의사결정 및 수익률을 극대화하는 주식 매매의 목적을 이루기에 한계가 있음
- 불확실한 환경에서 매매를 해야하는 현실의 문제에 맞도록 에이전트가 수 십만개의 시나리오를 통해 수익률을 극대화할 수 있는 액션을 학습하게하여 수익률을 극대화하고자 함

## 방법론
- 정책 경사를 활용하여 Policy Network의 액션(Buy, Sell, Hold) 학습
- 보상함수 설계
  - 수익률 발생 시 Discrete한 보상(+1, -1) 대신 수익률 자체를 Reward로 지급하여 수익률을 극대화 할 수 있는 방안 탐색
  - Monte carlo, Target Difference 방식 중 주식 거래 도메인에 더 효과적인 방식 채택

## 개발 방향

### 데이터 및 처리
- KOSPI 200에 해당하는 종목의 지난 10년(2013.01.01~2022.12.31) 일별 데이터를 수집(2023.08.26 기준)
  - 수집 데이터 : 시가, 고가, 저가, 종가, 거래량, 투자자별 거래대금(기관, 개인, 외국인)
- 거래량 및 가격 지표를 생성 및 전처리(클렌징, 변수변환 등)

### 환경 구성
- Environment와 state를 구성하기 위한 작업 실시
  - Environment : 개별 종목별 데이터
  - State : Agent가 볼 수 있는 특정 시점 t의 데이터
- Policy Network와 Agent의 거래를 위한 학습 데이터와 차트 데이터 파이프라인 구축

### 에이전트
- 거래 관련 파라미터 선언
  - 운용자금, 포트폴리오 가치, 평단가, 거래 수수료 등
- 거래 관련 함수 선언
  - Buy, Sell, Hold 등
- Policy Network 최적화 함수 선언
