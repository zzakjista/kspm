import torch 

class Trainer:

    def __init__(self,  agent, args):
        self.agent = agent
        self.device = args.device
        self.max_episode = 1 #args.max_episode
        self.batch_size = args.batch_size
        # self.len = len(self.env.close) - 1 # data의 총 길이
        self.done = False  # !! env의 done 쓸 지 trainer에서 done 쓸 지 agent의 steps_done도 컨트롤 해야됨 !! 
        self.agent.env.stock_code = '373220'


    def train(self):
        self.agent.env.state = self.agent.env.state_data['train']
        self.agent.env.chart = self.agent.env.chart_data['train']
        self.len = self.agent.env.state.shape[0] - 1
        # 일단은 에피소드를 들어가기전에 환경과 에이전트 세팅까지 되어있음 # 

        # for e in range(self.max_episode):
        #    self.run_episode()
           
            
            # !! model 저장 정책 !!
            # 1. recent_model.pt를 일정 episode마다 저장
            # 2. val에서 profit이 제일 좋았던 policy net 저장

            # if e % 10 == 0:
            #     torch.save(self.agent.policy_net, 'policy_net.pt')
            #     print('model is saved')

    def evaluate(self, test_mode):
        if test_mode == 'val':
            self.agent.env.state = self.agent.env.state_data['val']
            self.agent.env.chart = self.agent.env.chart_data['val']
        if test_mode == 'test':
            self.env.state = self.env.state_data['test']
            self.env.chart = self.env.chart_data['test']
        self.len = self.agent.env.state.shape[0] - 1
        # self.run_episode()
        # !!train_one_episode !!

    def run_episode(self): # only one episode
        # 초기화 
        self.agent.env.reset()      
        self.done = False
        self.agent.reset()
        total_profit = 0

        for t in range(self.len):
            
            state = self.agent.env.get_state()
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            action = self.agent.select_action(state)
            next_state = self.agent.env.get_next_state()
            reward = 0 
            profitloss = self.agent.act(action, 0.5)
            reward += profitloss
            self.agent.memory.push(state, action, reward, next_state, self.done) # !! 언제 push할 지 정해야함 !!
            self.agent.env.current_idx += 1
            self.agent.env.state_idx += 1
            if t == self.len-1: # 마지막 state 이전 state에서 done = True
                self.done = True
            
            if len(self.agent.memory) > self.batch_size:
                experiences = self.agent.memory.sample(self.batch_size)
                # !! self.agent.optimize_model(experiences) !! 
                
            if self.done:
                print('완료')
                print('total_profit:', (self.agent.portfolio_value-self.agent.initial_balance)/self.agent.initial_balance)
                print('buy_cnt:', self.agent.buy_cnt)
                print('sell_cnt:', self.agent.sell_cnt)
                print('hold_cnt:', self.agent.hold_cnt)
                break
    
    
        
    