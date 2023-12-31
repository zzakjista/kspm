import torch
import torch.nn as nn

class VALSTM(nn.Module) :
    def __init__(self, num_classes, args):
        super(VALSTM, self).__init__()
        self.device = args.device
        self.num_classes = num_classes
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        # self.dropout = nn.Dropout(args.dropout)
        self.lstm_layers = nn.LSTM(input_size=self.input_size, 
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=args.dropout,
                                   batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        # orthogonal initialization
        self._initialize()


            
    def _initialize(self):
        for name, param in self.lstm_layers.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            else:
                raise ValueError
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x) :
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, (hn, cn) = self.lstm_layers(x, (h_0, c_0))
        # out[:,-1,:] : (batch_size, seq_length, hidden_size) -> (batch_size, hidden_size)
        # hn[-1,:,:] : (num_layers, batch_size, hidden_size) -> (batch_size, hidden_size)
        out = self.fc(hn[-1, :, :]) 
        return out