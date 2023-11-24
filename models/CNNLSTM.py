import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, args):
        super(CNNLSTM, self).__init__()
        self.in_channel = args.in_channel
        self.out_channel = args.out_channel
        self.num_classes = num_classes
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = nn.Dropout(args.dropout)
        self.conv1d_1 = nn.Conv1d(in_channels=self.in_channel,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=16,
                                out_channels=self.out_channel,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        
        self.lstm = nn.LSTM(input_size=self.out_channel, 
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=args.dropout,
                                   batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        # orthogonal initialization
        self._initialize()

    def _initialize(self):
        for name, param in self.lstm.named_parameters():
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

    def forward(self, x):
   # Raw x shape : (B, S, F) => (B, 10, 3)
        
        # Shape : (B, F, S) => (B, 3, 10)
        x = x.transpose(1, 2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        x = self.conv1d_1(x)
        # Shape : (B, C, S) => (B, 32, 10)
        x = self.conv1d_2(x)
        # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters() # GPU에서 가중치를 로드할 때 메모리를 효율적으로 사용하게 해줌
        # Shape : (B, S, H) // H = hidden_size => (B, 10, 50)
        _, (hidden, _) = self.lstm(x)
        # Shape : (B, H) // -1 means the last sequence => (B, 50)
        x = hidden[-1]
        
        # Shape : (B, H) => (B, 50)
        x = self.dropout(x)
        
        # Shape : (B, 32)
        x = self.fc(x)
        # Shape : (B, O) // O = output => (B, 1)
        #x = self.fc_layer2(x)

        return x