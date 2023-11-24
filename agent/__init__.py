from .Agent import A2C
from .ReplayMemory import ReplayMemory

def agent_factory(environment, policy_net, critic_net, args):
    memory = ReplayMemory(args.capacity)
    agent = A2C(environment, policy_net, critic_net, memory, args)
    return agent