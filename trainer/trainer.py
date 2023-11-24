import torch 
import pandas as pd
from path_control import path_master
from pathlib import Path
from logger import *
from config import STATE_DICT_KEY , OPTIMIZER_STATE_DICT_KEY
import pickle
import time
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, agent, experiment_path, args):
        self.args = args
        self.agent = agent
        self.device = args.device
        self.experiment_path = experiment_path
        self.is_parallel = False
        self.best_metric_key = 'Profit'
        self.all_episode = 0
        self.max_episode = args.max_episode 
        self.batch_size = args.batch_size
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.start_time_log = None
        
        if args.pretrained:
            self.load_pretrained_model()
    
    def run(self):
        stocks = self.load_train_stock_list()
        self.start_time_log = time.time()
        for i, stock in enumerate(stocks):      
            self.train(stock)
            print(f'{round((i+1)/len(stocks)*100,0)}% stock done')
        if input('test를 진행하시겠습니까?(y/n):') == 'y':
            print('test를 진행합니다.')
            self.run_test_only()
    
    def run_test_only(self):
        stocks = list(self.agent.env.dataset.keys())
        for stock in stocks:
            test_result = {}
            test_profit, history = self.test(stock)
            test_result[stock] = {'test_profit': test_profit, 'history': history}
            self.save_test_result(test_result)
            print(f'{stock} --- test_profit: {test_profit}')
        
    # 종목마다 log를 새로 쌓는다?
    def train(self, stock):
        self.agent.env.config_environment(stock)
        self.val_loggers[-1].best_metric = -0.1 # 종목별 best metric 초기화 0으로 하게되면 손해 볼 때 학습이 안되니 -inf or 적절한 손절 범위 -0.1?
        self._set_train_env()
        self.agent.temperature = self.agent.init_temperature
        self.agent.max_trading_unit = self.agent.balance // self.agent.env.current_price()
        print(f'switch stock to {stock}, initial metric and temp')
        for e in range(self.max_episode):
            train_profit, policy_loss, critic_loss = self.run_episode('train')
            self.all_episode += 1
            self.agent.temperature *= self.agent.temperature_decay # exploratory decay each episode
            log_data = {
                    'state_dict': (self._create_state_dict()),
                    'policy_loss': policy_loss,
                    'critic_loss': critic_loss,
                    'episode' : self.all_episode
            }
            self.logger_service.log_train(log_data)
            # 현재 시간 - 시작 시간
            time_log = time.time() - self.start_time_log
            # 시,분,초
            print('running: ',time.strftime('%H:%M:%S', time.gmtime(time_log)))
            print(f'{(e+1)/self.max_episode*100:.2f}% done ----- train_profit: {train_profit}')
            print(f'policy_loss: {policy_loss}, critic_loss: {critic_loss}\n')
            print('-'*30)
            if e % 1 == 0: # e % update_interval == 0
                val_profit, _, _ = self.evaluate('val',stock,e)
                print(f'val_profit: {val_profit}\n')
            self._set_train_env()
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()
        return train_profit, policy_loss, critic_loss
    
    def evaluate(self, test_mode, stock, e):
        self._set_evaluate_env(test_mode)
        with torch.no_grad():            
            val_profit, policy_loss, critic_loss = self.run_episode(test_mode)  
        log_data ={
            'state_dict': (self._create_state_dict()),
            'Profit': val_profit,
            'stock_code': stock,
            'episode': e + 1
        }
        self.logger_service.log_val(log_data) 
        return val_profit, policy_loss, critic_loss

    def test(self, stock):
        self.agent.env.config_environment(stock)
        self._set_evaluate_env('test')
        with torch.no_grad():
            test_profit, _, _ = self.run_episode('test')
            history = pd.DataFrame(self.agent.history)
        return test_profit, history
    
    def inference(self, stock): # 오늘 날짜에 대한 state만 가져와서 action을 선택
        self.agent.env.config_environment(stock)
        self._set_evaluate_env('test')
        with torch.no_grad():
            state = self.agent.env.get_last_state()
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0) # (1, sequence_length, feature_size)
            action, prob = self.agent.select_action(state)
        return action

    def run_episode(self, mode): 
        self.agent.env.reset()      
        self.agent.reset()
        total_profit = 0
        total_policy_loss = 0
        total_critic_loss = 0 
        for t in range(self.len):
            reward = 0 
            state = self.agent.env.get_state()
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0) # (1, sequence_length, feature_size)
            # policy # 
            action, prob = self.agent.select_action(state, mode)
            next_state = self.agent.env.get_next_state()
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            action, action_prob = self.agent.act(action, prob)

            next_stock_value = self.agent.env.next_price() * self.agent.num_stocks #+ self.agent.balance
            next_portfolio_value = next_stock_value + self.agent.balance
            # critic #
            state_value = self.agent.critic_net(state) # critic_net에서 state_value 추정
            next_state_value = self.agent.critic_net(next_state)
            
            if t == self.len-1: # 마지막 state 이전 state에서 done = True
                self.agent.env.done = True

            if action == 0:
                reward += ((next_portfolio_value - self.agent.portfolio_value) / self.agent.portfolio_value)
            elif action == 1:
                reward += (next_portfolio_value - self.agent.portfolio_value) / self.agent.portfolio_value \
                    + (self.agent.env.current_price() - self.agent.avg_buy_price) / self.agent.avg_buy_price 
            else:
                reward += (next_portfolio_value - self.agent.portfolio_value) / self.agent.portfolio_value
            self.agent.memory.push(state_value, action_prob, reward, next_state_value, self.agent.env.done)

            # optimize model and clear batch #
            if len(self.agent.memory) == self.batch_size:
                experiences = self.agent.memory.pop(self.batch_size)
                policy_loss, critic_loss = self.agent.caculate_loss(experiences)
                if mode == 'train':
                    self.agent.optimize_model(policy_loss, critic_loss)
                total_policy_loss += policy_loss.item()
                total_critic_loss += critic_loss.item()
            
            if mode == 'test':
                self.agent.history['date'].append(self.agent.env.current_date())
                self.agent.history['price'].append(self.agent.env.current_price())
                self.agent.history['action'].append(action)
            
            if self.agent.env.done:
                total_profit = (self.agent.portfolio_value-self.agent.initial_balance)/self.agent.initial_balance
                print('buy_cnt:', self.agent.buy_cnt)
                print('sell_cnt:', self.agent.sell_cnt)
                print('hold_cnt:', self.agent.hold_cnt)
                return total_profit, total_policy_loss, total_critic_loss
                
            self.agent.env.current_idx += 1
            self.agent.env.state_idx += 1

    def _set_train_env(self):
        self.agent.env.state = self.agent.env.state_data['train']
        self.agent.env.chart = self.agent.env.chart_data['train']
        self.len = self.agent.env.state.shape[0] - 1
        self.agent.policy_net.train()
        self.agent.critic_net.train()

    def _set_evaluate_env(self, test_mode):
        self.agent.policy_net.eval()
        self.agent.critic_net.eval()
        if test_mode == 'val':
            self.agent.env.state = self.agent.env.state_data['val']
            self.agent.env.chart = self.agent.env.chart_data['val']
            
        if test_mode == 'test':
            self.agent.env.state = self.agent.env.state_data['test']
            self.agent.env.chart = self.agent.env.chart_data['test']
        self.len = self.agent.env.state.shape[0] - 1

    def save_test_result(self, test_result):
        root = Path(self.experiment_path)
        test_result_path = root.joinpath('test_result.pickle')
        if not os.path.exists(test_result_path):
            with open(test_result_path, 'wb') as f:
                pickle.dump(test_result, f)
        else:
            with open(test_result_path, 'rb') as f:
                test_results = pickle.load(f)
                test_results.update(test_result)
                with open(test_result_path, 'wb') as f:
                    pickle.dump(test_results, f)
        print('test_result is saved')
    
    def load_train_stock_list(self):
        all_stock_list = list(self.agent.env.dataset.keys())
        if self.args.pretrained:
            if input('오류로 인해 이어서 학습하시겠습니까?(y/n):') == 'y':
                stock_code = input('이어서 학습할 종목코드를 입력하세요: ')
                stock_list = all_stock_list[all_stock_list.index(stock_code):]
                print(f'남은 {len(stock_list)}개의 종목을 학습합니다.')
                return stock_list
            else:
                print('새로 학습합니다.')
                stock_list = all_stock_list
        else:
            stock_list = all_stock_list
        return stock_list

    def _create_loggers(self):
        root = Path(self.experiment_path)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='policy_loss', graph_name='policy_loss', group_name='Train'),
            MetricGraphPrinter(writer, key='critic_loss', graph_name='critic_loss', group_name='Train'),
        ]

        val_loggers = []
        val_loggers.append(
                MetricGraphPrinter(writer, key='Profit', graph_name='Profit', group_name='Profit'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric_key))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: {'policy_net' : self.agent.policy_net.module.state_dict() if self.is_parallel else self.agent.policy_net.state_dict(),
                             'critic_net' : self.agent.critic_net.module.state_dict() if self.is_parallel else self.agent.critic_net.state_dict()},
            OPTIMIZER_STATE_DICT_KEY: self.agent.optimizer.state_dict(),
        }

    def load_pretrained_model(self):
        pm = path_master(self.args)
        pretrained_model_path = pm.get_pretrained_folder_path(self.args).joinpath('models').joinpath('recent_model.pth')
        models = torch.load(pretrained_model_path)
        model_state = models['model_state_dict']
        self.agent.policy_net.load_state_dict(model_state['policy_net'])
        self.agent.critic_net.load_state_dict(model_state['critic_net'])
        self.agent.optimizer.load_state_dict(models['optimizer_state_dict'])
        print('pretrained model is loaded')

        

        
    
    
        
    