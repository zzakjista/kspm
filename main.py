from utils import *
from dataset import dataset_factory
from env import env_factory
from models import model_factory
from agent import agent_factory
from trainer import trainer_factory
from parse import args

def main():
    experiment_path = setup_service(args)
    dataset = dataset_factory(args)
    env = env_factory(dataset, args)
    policy_net, critic_net = model_factory(args)
    agent = agent_factory(env, policy_net, critic_net, args)
    trainer = trainer_factory(agent, experiment_path, args)
    if input('Are you sure to train model?: (y/n)') =='y':
        trainer.run()
    elif input('Or test model by loading pretrained model?: (y/n)') == 'y':
        trainer.run_test_only()
    else:
        print('exit the service')

if __name__ == "__main__":
    main()

    
### 주요 기능 정리 ###
# args.template == 'train' -> train, test <KOSPI200의 내장된 테스트 데이터셋>
# args.template == 'test' -> test <주식과 일자를 불러와서 테스트 가능>
# args.template == 'inference -> inference <주식의 당일 state에 대한 action만 배출> -> 우선순위 낮음