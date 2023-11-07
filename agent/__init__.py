from .Agent import A2C

def agent_factory(environment, policy_net, critic_net, args):
    agent = A2C(environment, policy_net, critic_net, args)
    return agent