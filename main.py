from dataset import dataset_factory
from env import env_factory
from agent.Agent import Agent
from agent.LSTM import LSTM # 모델 끌어오는 pipeline을 agent와 독립시켜도됨
from agent.ReplayMemory import ReplayMemory
from trainer import Trainer
from parse import args
import pandas as pd 

def main():

    dataset = dataset_factory(args)
    env = env_factory(dataset, args)
    return env
    # return print(env.chart_data['train'].head())
    # model = LSTM(17, args.hidden_size, args.num_layers, args.action_kind).to(args.device)
    # replay_memory = ReplayMemory(args.capacity) # ReplayMemory 삭제 예정
    # agent = Agent(env, model, replay_memory,args)
    # trainer = Trainer(agent,args)
    # trainer.train()

if __name__ == "__main__":
    main()
    
