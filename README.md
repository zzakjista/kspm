# Advantage Actor Critic TRADER IN KOSPI 200

## 개요
- 강화학습 에이전트가 수익률을 극대화하는 거래를 할 수 있도록 학습하고 KOSPI 200 종목에서 실전 매매를 진행

## 배경 
- 딥러닝을 이용하여 주식 가격을 예측하는 프로젝트는 일반적으로 시도되었으나, 단순 예측에 기반하기 때문에 투자 의사결정 및 수익률을 극대화하는 주식 매매의 목적을 이루기에 한계가 있음
- 불확실한 환경에서 매매를 해야하는 현실의 문제에 맞도록 에이전트가 수 십만개의 시나리오를 통해 수익률을 극대화할 수 있는 액션을 학습하게하여 수익률을 극대화하고자 함

## 방법론 : Advantage Actor Critic 
- 정책 경사를 활용하여 Policy Network의 액션(Buy, Sell, Hold) 학습
- Critic Network 도입으로 Policy Network가 예측한 action의 가치 측정
- Advantage Term (Q(st,at)- V(st))를 이용하여 Policy, Critic Network 학습
- A2C는 Onpolicy이므로 N개에 대한 Batch 학습을 진행하고 샘플을 재사용하지 않는다
- 보상함수 설계
  - one buy - one sell이 일어날 시 매도 차익에 대한 수익률로 보상 설정 
  - Monte carlo, Target Difference 방식 중 주식 거래 도메인에 더 효과적인 방식 채택

## 개발 방향

### 데이터 및 처리
- KOSPI 200에 해당하는 종목의 지난 10년(2013.01.01~2022.12.31) 일별 데이터를 수집(2023.08.26 기준)
  - 수집 데이터 : 시가, 고가, 저가, 종가, 거래량, 투자자별 거래대금(기관, 개인, 외국인)
- 거래량 및 가격 지표를 생성 및 전처리(클렌징, 변수변환 등)
- 학습기간 동안 사용한 스케일러를 테스트 때 사용하기 위해서 종목별 스케일러 저장

### 환경 구성
- Environment와 state를 구성하기 위한 작업 실시
  - 개별 종목 코드를 부르면 전체 주가 데이터를 불러오는 기능
  - 환경에는 두 가지 데이터 셋이 존재
    - 차트 데이터 : 에이전트가 실거래가로 매매하기 위해 사용 (시가, 저가, 고가, 종가)
    - 학습 데이터 : policy network에 들어가기 위한 데이터 (각종 지표들)
  - 지정한 윈도우 사이즈만큼 개별 state를 호출하고 next state를 불러주는 기능

### 에이전트
- 거래 관련 파라미터
  - 운용자금, 포트폴리오 가치, 평단가, 거래 수수료 등
- 거래 관련 함수 선언
  - Buy, Sell, Hold 등
- Policy Network (Gradient Ascent)
  - loss = -logprob * advantage
- Critic Network (Gradient Descent)
  - loss = (advantage)^2
- Advantage = V(st+1) - V(st) :
  - Q(st,at)를 bellman equation으로 변환 후 state만을 가지고 advantage term을 나타낼 수 있어 연산 효율화 가능

