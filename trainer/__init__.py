from .trainer import Trainer

def trainer_factory(agent, experiment_path,  args):
    trainer = Trainer(agent, experiment_path, args)
    return trainer