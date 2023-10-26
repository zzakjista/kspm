from .environment import Env

def env_factory(data, args):
    return Env(data, args)
