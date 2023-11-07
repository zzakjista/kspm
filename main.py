from dataset import dataset_factory
from env import env_factory
from models import model_factory
from agent import agent_factory
from trainer import trainer_factory
from parse import args
import pandas as pd 

def main():
    dataset = dataset_factory(args)
    env = env_factory(dataset, args)
    policy_net, critic_net = model_factory(args)
    agent = agent_factory(env, policy_net, critic_net, args)
    trainer = trainer_factory(agent, args)
    return trainer
    # return print(env.chart_data['train'].head())
    # model = LSTM(17, args.hidden_size, args.num_layers, args.action_kind).to(args.device)
    # replay_memory = ReplayMemory(args.capacity) # ReplayMemory 삭제 예정
    # agent = Agent(env, model, replay_memory,args)
    # trainer = Trainer(agent,args)
    # trainer.train()

if __name__ == "__main__":
    main()
    
