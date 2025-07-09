import torch 
import torch.nn as nn
import torch.nn.functional as F





class Bahdanau_Attention(nn.Module):
    def __init__(self,model_size,atten_size):
        super().__init__()
        self.W_encode = nn.Linear(model_size,atten_size,bias=False)
        self.W_decode = nn.Linear(model_size,atten_size,bias=False)
        self.V = nn.Linear(atten_size,1,bias=False)
    

    def forward(self,encode_state,decode_hid_state):

        decode_hid_state = decode_hid_state.unsqueeze(1)

        scores = self.V(F.tanh(self.W_encode(encode_state)+self.W_decode(decode_hid_state)))
        probs = F.softmax(scores,dim=1)
        context_vector = torch.sum(probs*encode_state,dim = 1)
      
        return scores,context_vector









class Luong_Dot_Attention(nn.Module):
    def __init__(self,model_size,atten_size):
        super().__init__()
    
    
    def forward(self,encode_state,hid_state):

        hid_state = hid_state.unsqueeze(2)


        scores = torch.matmul(encode_state,hid_state).squeeze(-1)
        probs = F.softmax(scores, dim = 1)
        context_vector = torch.sum(probs.unsqueeze(2)*encode_state,dim = 1)

        return scores,context_vector










class Luong_Gen_Attention(nn.Module):
    def __init__(self,model_size,atten_size):
        super().__init__()
        self.W = nn.Linear(model_size,atten_size)

    def forward(self,encode_state,hid_state):


        encode_state_multiplied = self.W(encode_state)
        hid_state = self.W(hid_state).unsqueeze(2)

        scores = torch.matmul(encode_state_multiplied,hid_state).squeeze(-1)
        probs = F.softmax(scores,dim =1)
        context_vector = torch.sum(probs.unsqueeze(2)*encode_state, dim = 1)
        return scores, context_vector





# class Luong_Concat_Attention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.W = nn.Linear(embed_dim,embed_dim,bias=False)
#         self.v = nn.Linear(embed_dim,embed_dim)

#     def forward(self,encode_stae,decode_state):


#         return probs, context_vector


class Luong_Concat_Attention(nn.Module):
    def __init__(self, model_size, atten_size):
        super().__init__()
        self.W = nn.Linear(model_size * 2, atten_size, bias=False)  
        self.v = nn.Linear(atten_size, 1, bias=False)

    def forward(self, encode_state, decode_state):
       
        seq_len = encode_state.size(1)
        decode_expanded = decode_state.unsqueeze(1).repeat(1, seq_len, 1) 

       
        concat = torch.cat((encode_state, decode_expanded), dim=2)

      
        energy = torch.tanh(self.W(concat))           
        scores = self.v(energy).squeeze(-1)            
        probs = F.softmax(scores, dim=1)              
        context_vector = torch.sum(probs.unsqueeze(2) * encode_state, dim=1) 

        return scores, context_vector








def get_attention_class(name):
    return {
        "Bahdanau": Bahdanau_Attention,
        "LuongDot": Luong_Dot_Attention,
        "LuongGeneral": Luong_Gen_Attention,
        "LuongConcat": Luong_Concat_Attention
    }[name]
