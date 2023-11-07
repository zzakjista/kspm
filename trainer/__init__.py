from .trainer import Trainer

def trainer_factory(agent, args):
    trainer = Trainer(agent, args)
    return trainer