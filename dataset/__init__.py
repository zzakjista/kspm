from .preprocess import Preprocesser
from collect import collect_stock

def dataset_factory(args):
    ppc = Preprocesser(args)
    if args.template == 'train':
        data = ppc.make_dataset()
 
    elif args.template == 'test':
        data = collect_stock() # 실시간 수집해서 
        data = ppc.preprocess(data)
    return data
