import torch 
import torch.nn as nn
import torch.nn.functional as F


class Vanilla_lstm(nn.Module):
    def __init__(self,embed_dim,model_size,output_size):
        super(Vanilla_lstm,self).__init__()
        
        self.model_size = model_size        
        self.lstm = nn.LSTM(embed_dim,model_size,batch_first=True)
        self.fl = nn.Linear(model_size,output_size)

    def forward(self,x):
        
        ho = torch.zeros(1,x.shape[0],self.model_size).to(x.device)
        co = torch.zeros(1,x.shape[0],self.model_size).to(x.device)

        out, (hn,cn) = self.lstm(x,(ho,co))        

        return out,hn[-1]









        
class Bidirectional_lstm(nn.Module):
    def __init__(self,embed_dim,model_size,output_size):
        super(Bidirectional_lstm,self).__init__()
        
        self.model_size = model_size        
        self.lstm = nn.LSTM(embed_dim,model_size,batch_first=True,bidirectional=True)
        self.fl = nn.Linear(model_size*2,output_size)

    def forward(self,x):
        
        ho = torch.zeros(2,x.shape[0],self.model_size).to(x.device)
        co = torch.zeros(2,x.shape[0],self.model_size).to(x.device)

        out, (hn,cn) = self.lstm(x,(ho,co))
        final_hn = torch.cat((hn[-2], hn[-1]), dim=1)
        
        return out,final_hn








class Vanilla_rnn(nn.Module):
    def __init__(self,embed_dim,model_size,output_size):
        super(Vanilla_rnn,self).__init__()
        
        self.model_size = model_size
        self.rnn = nn.RNN(embed_dim,model_size,batch_first=True)
        self.fl = nn.Linear(model_size,output_size)
    
    def forward(self,x):
        ihs = torch.zeros(1,x.shape[0],self.model_size).to(x.device)
        out,hs = self.rnn(x,ihs)
        
        return out,hs[-1]
    





    


class Bidirectional_rnn(nn.Module):
    def __init__(self,embed_dim,model_size,output_size):
        super(Bidirectional_rnn,self).__init__()
        
        self.model_size = model_size
        self.rnn = nn.RNN(embed_dim,model_size,batch_first=True,bidirectional=True)
        self.fl = nn.Linear(model_size*2,output_size)
    
    def forward(self,x):
        ihs = torch.zeros(2,x.shape[0],self.model_size).to(x.device)
        out,hs = self.rnn(x,ihs)
        
        final_hs = torch.cat((hs[-2],hs[-1]),dim = 1)

        return out,final_hs







def get_model_class(name):
    return {
        "VanillaRNN": Vanilla_rnn,
        "VanillaLSTM": Vanilla_lstm,
        "BiRNN": Bidirectional_rnn,
        "BiLSTM": Bidirectional_lstm
    }[name]




