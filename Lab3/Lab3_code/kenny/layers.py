import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads=num_heads
        self.dim=dim
        self.d_k=dim//num_heads
        self.attn_drop=attn_drop

        self.query_linear=nn.Linear(dim,dim)
        self.key_linear=nn.Linear(dim,dim)
        self.value_linear=nn.Linear(dim,dim)
        self.drop=nn.Dropout(attn_drop)
        self.proj_linear=nn.Linear(dim,dim)
    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        B,T,_=x.shape
        #Q K V
        query=self.query_linear(x)
        key=self.key_linear(x)
        value=self.value_linear(x)
        #att(Q,K,V)
        query=query.reshape(B,T,self.num_heads,self.d_k).transpose(1,2)
        key=key.reshape(B,T,self.num_heads,self.d_k).transpose(1,2)
        value=value.reshape(B,T,self.num_heads,self.d_k).transpose(1,2)
        #Q @ K^T / sqrt(dk)
        atten=query @ key.transpose(2,3)#[B,16,T,T]
        atten =atten/math.sqrt(self.d_k)
        #softmax
        atten=atten.softmax(dim=-1)
        atten=self.drop(atten)
        x=atten @ value#[B,16,T,T] @[B,16,T,48]->[B,16,T,48]
        #[B,16,T,48]
        x=x.transpose(1,2).reshape(B,T,self.dim)
        x=self.proj_linear(x)
        return x

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
