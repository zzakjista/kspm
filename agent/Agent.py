import torch
import torch.nn as nn
import numpy as np
import random  
import math 

class A2C:

    def __init__(self, environment, policy_net, critic_net, args): # replay memory가 필요없을 것으로 판단
        self.device = args.device
        self.env = environment
        # policy & critic  #
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.action_kind = args.action_kind     
        self.policy_net = policy_net
        self.critic_net = critic_net

        # train #
        self.gamma = args.gamma
        self.lr = args.lr
        self.batch_size = args.batch_size

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)
        self.loss_critic = nn.MSELoss()

        # self.steps_done = 0
        # self.eps_start = args.eps_start
        # self.eps_end = args.eps_end
        # self.eps_decay = args.eps_decay

        # trade #
        self.initial_balance = args.initial_balance # 초기 자본금
        self.balance = args.initial_balance # 현재 현금 잔고
        self.num_stocks = 0 # 보유 주식 수
        self.portfolio_value = 0 # 포트폴리오 가치
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익

        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율
        self.avg_buy_price = 0  # 평단가

        self.min_trading_unit = args.min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = args.max_trading_unit # 최대 단일 거래 단위

        self.buy_cnt = 0
        self.sell_cnt = 0
        self.hold_cnt = 0
        # 매매 수수료 및 세금
        self.trading_charge = 0.00015  # 거래 수수료 0.015%
        self.trading_tax = 0.0025  # 거래세 0.25%
        
    def set_balance(self, balance):
        self.initial_balance = balance

    def reset(self):
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_stocks = 0
        self.buy_cnt = 0
        self.sell_cnt = 0
        self.hold_cnt = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0
        self.avg_buy_price = 0 

    def get_states(self): #에이전트의 현재 상태를 반환
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.env.current_price())
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value # 포트폴리오 가치 비율
        )
        if self.avg_buy_price > 0:
            self.stock_profitloss = (
                (self.env.current_price() - self.avg_buy_price) / self.avg_buy_price
            ) # 현재 가격 대비 평단가의 손익률
        else:
            self.stock_profitloss = 0
        return (self.ratio_hold, self.ratio_portfolio_value, self.stock_profitloss)

    def select_action(self, state):
        # sample = random.random() ----> 이 
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        #                 math.exp(-1. * self.steps_done / self.eps_decay) 
        # # eps_threshold : end + (start - end) * exp(-1 * steps / decay) 가 0에 가까워질수록 랜덤 액션을 취할 확률이 줄어듬
        # self.steps_done += 1 # steps_done이 커질수록 eps_threshold가 작아짐
    

        # DEV : A2C에서는 epsilon greedy 탐험 방식은 a2c에서 사용하진 않음, Softmax의 temperature을 조절하는 방식으로 탐험을 하는 방법이 있음 
        # -> 노션 백로그 참조 https://www.notion.so/24c2c61d5fe84ccc97048c48f8e6cd33
        # Softmax T : exp(x/T) / sum(exp(x/T)) -> T가 작아질수록 랜덤 액션을 취할 확률이 줄어듬
        # if sample > eps_threshold: # 랜덤 액션을 취할 확률을 줄여나감

        with torch.no_grad():
            self.policy_net.eval() 
            prob = self.policy_net(state) # 정책 신경망으로 행동을 예측 각 행동에 대한 확률 
            action = pred.max(1)[1].view(1, 1) # 행동 중 가장 큰 값의 인덱스를 가져옴 Softmax. 훈련 중엔 max값이 아닌 확률에 의해 sampling
            self.policy_net.train()
        return action #.max(1)[1]
        # else:
        #     return torch.tensor([[random.randrange(self.action_kind)]], device=self.device, dtype=torch.long) # random action 

    # 훈련 중엔 거래 관련 제약을 적용하는 것이 좋지않아보임, policy가 예측한대로 행동하게끔 하고, 훈련이 끝난 후에 제약을 걸어주는 것이 좋을 듯함
    # 즉 제약을 키고 끌 수 있게 기능 개발 필요
    def validate_action(self, action): # buy or sell이 가능한지 확인
        if action == 0:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.env.current_price() * (
                1 + self.trading_charge) * self.min_trading_unit:
                return False
        elif action == 1:
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence): # 현재 1주 거래 룰 -> confidence에 따라서 거래 단위를 결정하는 방안도 좋을 듯함 
        if np.isnan(confidence): # confidence가 nan이면 최소 거래 단위만큼 거래
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit - 
                self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):
        if not self.validate_action(action): # validate_action에서 제한사항이 생기면 hold
            action = 2
        curr_price = self.env.current_price()
        trading_unit = self.decide_trading_unit(confidence)
        if action == 0:
            balance = self.balance - curr_price * (1 + self.trading_charge) * trading_unit
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (curr_price * (1 + self.trading_charge))), # 최대 구매 가능 수량
                    self.max_trading_unit
                ), self.min_trading_unit)
            self.buy(curr_price, trading_unit)

        elif action == 1:
            trading_unit = min(trading_unit, self.num_stocks)
            self.sell(trading_unit)
        
        elif action == 2:
            self.hold()
        
        self.portfolio_value = self.balance + curr_price * self.num_stocks # 잔고 + 주식 가치
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) / self.initial_balance # 초기 자본금 대비 손익률 갱신
        )
        return self.profitloss

    # 거래정책 : 일단 전량 매도/매수가 가능하게 함
    def buy(self, price, trading_unit): # 원하는 수량만큼 매수 
        buy_cost =  price * (1 + self.trading_charge) * trading_unit # 매수 비용
        if buy_cost > 0:
            self.avg_buy_price = (self.avg_buy_price * self.num_stocks + price) / (self.num_stocks + trading_unit)  # 주당 매수 단가 갱신
            self.balance -= buy_cost  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.buy_cnt += 1  # 매수 횟수 증가
            print("Buy: " + str(round(buy_cost,0)))
    
        
    def sell(self, trading_unit): # 원하는 수량만큼 매도
        profit = self.env.current_price() * (1 - (self.trading_charge + self.trading_tax)) * trading_unit
        if profit > 0:
            # self.avg_buy_price = (self.avg_buy_price * self.num_stocks - curr_price) / (self.num_stocks - trading_unit) if self.num_stocks > trading_unit else 0  # 평단가 매도 시 영향 안줄텐데
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += profit  # 보유 현금을 갱신
            self.sell_cnt += 1  # 매도 횟수 증가
            print("Sell: " + str(round(profit,0)))
        
    
    def hold(self):
        self.hold_cnt += 1 
        print("Hold")


    def optimize_model(self, action_pred):
        pass 
        # optimize_model 개발 사항
        # guideline
        # 1. state를 env로부터 받는다.
        # 2. state를 policy network에 넣어서 action과 prob을 뽑는다
        # 3. 정해진 액션에 따라 에이전트가 행동하고 reward를 얻는다 -> 이때 액션이 확정되었으니 에이전트의 자산 업데이트가 일어나야함
        # 4. env에서 next state를 받아온다
        # 5. state와 next state를 critic network에 넣어서 V(st+1)과 V(st)를 구한다
        # 6. V(st+1)과 V(st)를 이용해서 advantage를 구한다
        # 7. policy network의 loss를 구한다 -(log_prob * advantage) -> gradient descent라 앞에 -를 붙여줌
        # 8. critic network의 loss를 구한다 (V(st+1) - V(st))^2
        # 9. policy network와 critic network를 업데이트한다
        
        # 231019 메모
        # Policy Gradient 방식을 이용한다고 할 경우,
        # Gt를 구할 때 Monte carlo는 1 episode를 모두 진행한 후에 Gt를 구한다.
        # TD는 1 step 진행한 후에 Gt를 구한다. 
        # Object Function = E[log pi(a|s) * Gt]
        # pi(a|s) = policy network
        # Gt = reward + gamma * Gt+1
        # Gt or Q-value를 구하는 함수가 있어야한다 
        # 결국 Policy network말고도 Value network가 필요하다. -> A2C

        


