from .VALSTM import VALSTM
from .CNNLSTM import CNNLSTM

def model_factory(args):
    if args.model_name == 'VALSTM':
        policy_net = VALSTM(args.action_kind, args) # 같은 모델을 쓰되 action_kind만 다르게
        critic_net = VALSTM(args.value_kind, args)
    elif args.model_name == 'CNNLSTM':
        policy_net = CNNLSTM(args.action_kind, args)
        critic_net = CNNLSTM(args.value_kind, args)
    else:
        raise NotImplementedError
    return policy_net, critic_net