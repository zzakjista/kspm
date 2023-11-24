import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils
import numpy as np

class A2C:

    def __init__(self, environment, policy_net, critic_net, memory, args): 
        self.device = args.device
        self.env = environment
        self.memory = memory
        # policy & critic  #
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.action_kind = args.action_kind     
        self.policy_net = policy_net.to(self.device)
        self.critic_net = critic_net.to(self.device)
        # train #
        self.gamma = args.gamma
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.init_temperature = args.temperature
        self.temperature = args.temperature
        self.temperature_decay = args.temperature_decay
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.critic_net.parameters()), lr=self.lr) # combine two network's parameters
        self.loss_critic = nn.MSELoss()
        # trade #
        self.initial_balance = args.initial_balance 
        self.balance = args.initial_balance
        self.num_stocks = 0 
        self.portfolio_value = 0
        self.base_portfolio_value = 0 
        self.profitloss = 0 
        self.base_profitloss = 0 
        self.stock_value = 0
        self.ratio_hold = 0  
        self.ratio_portfolio_value = 0  
        self.avg_buy_price = 0  

        self.min_trading_unit = args.min_trading_unit  
        self.max_trading_unit = args.max_trading_unit         
        self.history = {'date':[],'price':[], 'action':[]}
        self.buy_cnt = 0
        self.sell_cnt = 0
        self.hold_cnt = 0

        self.trading_charge = 0.00015  # 거래 수수료 0.015%
        self.trading_tax = 0.0025  # 거래세 0.25%
        
    def set_balance(self, balance):
        self.initial_balance = balance

    def reset(self):
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.stock_value = 0
        self.num_stocks = 0
        self.history = {'date':[],'price':[], 'action':[]}
        self.buy_cnt = 0
        self.sell_cnt = 0
        self.hold_cnt = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0
        self.avg_buy_price = 0 

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.env.current_price())
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value 
        )
        if self.avg_buy_price > 0:
            self.stock_profitloss = (
                (self.env.current_price() - self.avg_buy_price) / self.avg_buy_price
            ) 
        else:
            self.stock_profitloss = 0
        return (self.ratio_hold, self.ratio_portfolio_value, self.stock_profitloss)

    def select_action(self, state, mode):
        prob = self.policy_net(state) 
        if self.num_stocks <= 0:
            prob[0][1] = -torch.inf
        elif self.balance < self.env.current_price() * (
            1 + self.trading_charge) * self.min_trading_unit:
            prob[0][0] = -torch.inf
        if mode == 'train':
            prob /= max(self.temperature+1,1) # add exploration noise : Softmax T : exp(x/T) / sum(exp(x/T))  # 1 is bias 
        prob = F.softmax(prob, dim=1) 
        print(prob)
        action = prob.multinomial(num_samples=1).item()
        prob = prob.squeeze(0)
        return action , prob

    def validate_action(self, action): 
        if action == 0:
            # check my balance
            if self.balance < self.env.current_price() * (
                1 + self.trading_charge) * self.min_trading_unit:
                return False
        elif action == 1:
            # check my stocks 
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence): 
        confidence = confidence.item()
        if np.isnan(confidence): 
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit - 
                self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):

        action_prob = confidence[action]
        curr_price = self.env.current_price()
        trading_unit = self.decide_trading_unit(action_prob)
        if action == 0:
            balance = self.balance - curr_price * (1 + self.trading_charge) * trading_unit
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (curr_price * (1 + self.trading_charge))), # maximum trading unit
                    self.max_trading_unit
                ), self.min_trading_unit)
            self.buy(curr_price, trading_unit)

        elif action == 1:
            trading_unit = min(trading_unit, self.num_stocks) # don't sell more than you have
            self.sell(trading_unit)
    
        elif action == 2:
            self.hold()

        self.portfolio_value =  self.balance + curr_price * self.num_stocks # update portfolio value 
        self.stock_value = curr_price * self.num_stocks
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) / self.initial_balance 
        )
        return action, action_prob

    def buy(self, price, trading_unit): 
        buy_cost =  price * (1 + self.trading_charge) * trading_unit
        if buy_cost > 0:
            self.avg_buy_price = (self.avg_buy_price * self.num_stocks + price * trading_unit) / (self.num_stocks + trading_unit)
            self.avg_buy_price = round(self.avg_buy_price, 0)
            self.balance -= buy_cost 
            self.num_stocks += trading_unit 
            self.buy_cnt += 1 
            # print("Buy: " + str(round(buy_cost,0)))
    
    def sell(self, trading_unit):
        profit = self.env.current_price() * (1 - (self.trading_charge + self.trading_tax)) * trading_unit
        if profit > 0:
            self.avg_buy_price = 0  if self.num_stocks < trading_unit else self.avg_buy_price  
            self.num_stocks -= trading_unit  
            self.balance += profit  
            self.sell_cnt += 1  
            # print("Sell: " + str(round(profit,0)))
    
    def hold(self):
        self.hold_cnt += 1 

    def caculate_loss(self, experiences):
        policy_loss = 0
        critic_loss = 0
        for experience in experiences:
            log_prob = torch.log(experience.prob)
            reward = experience.reward
            next_state_value = experience.next_state_value 
            next_state_value = next_state_value.detach() # fixed target value
            state_value = experience.state_value
            td_target = reward + self.gamma * next_state_value
            advantage = td_target - state_value
            policy_loss += self.policy_loss_calculate(log_prob, advantage)
            critic_loss += self.critic_loss_calculate(td_target, state_value)
        policy_loss /= len(experiences)
        critic_loss /= len(experiences)
        return policy_loss, critic_loss

    def optimize_model(self, policy_loss, critic_loss):
        loss = policy_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def policy_loss_calculate(self, log_prob, advantage):
        return -log_prob * advantage
    def critic_loss_calculate(self, td_target, state_value):
        return self.loss_critic(td_target, state_value)

        


