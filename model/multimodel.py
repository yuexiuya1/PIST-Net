#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from typing import Optional, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import testmodel
from . import resnet

class multi(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # self.mob=mobileone.mobileone(num_classes=5,variant='s1',with_class=False)
        self.mob=resnet.resnet18(num_classes=5, include_top=False)
        # self.mob=convnext.convnext_tiny(head=False)
        self.specmodel=testmodel.SPNet(with_class=False)
        self.multiout=nn.Sequential(
            # kan.KAN([16*(1917+80),100,5]),
            nn.Linear(512+30672, 1000),
            nn.ReLU(),
            # # nn.Dropout(p=0.5),
            nn.Linear(1000, 100),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            
            # kan.KAN([1000,100,5]),
            nn.Linear(100, 5),
            # nn.Linear(10,5),
            # nn.Sigmoid()
        )
    def forward(self,y: torch.Tensor,x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x = self.mob(x)
        x = x.view(x.size(0), -1)
        y = self.specmodel(y)
        y = y.view(y.size(0), -1)
        x = torch.cat((x,y),dim=1)
        x = self.multiout(x)
        return x
class multiTransformer(nn.Module):

    def __init__(self,
                 ) :
        super().__init__()
        self.mob=resnet.resnet18(num_classes=5, include_top=False)##
        # self.mob=convnext.convnext_tiny(head=False)
        # self.mob=mobileone.mobileone(num_classes=5,variant='s1',with_class=False)
        self.specmodel=testmodel.SPNet(with_class=False)
        self.multiout=MultiModalModel( key_value_dim=16, num_tokens=1917+32, d_model=16)
        
        

    def forward(self,y: torch.Tensor,x: torch.Tensor) -> torch.Tensor:
        x = self.mob(x)
        # x = x.view(x.size(0), -1)
        y = self.specmodel(y)
        # y = y.view(y.size(0), -1)
        # x = torch.cat((x,y),dim=1)
        x = self.multiout(x,y)
        return x

    
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(FixedPositionalEncoding, self).__init__()

        # Create constant positional encoding matrix with shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # Add positional encoding to input x
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiModalModel(nn.Module):
    def __init__(self, key_value_dim, num_tokens, d_model):
        super(MultiModalModel, self).__init__()
        
        # Linear layers to project A and B to the model dimension
        # self.piclayernormal=nn.LayerNorm((num_tokens-1917)*16)
        # self.piclayernormal2=nn.BatchNorm1d(1280)
        # self.speclayernormal=nn.LayerNorm([d_model,1917])
        # self.speclayernormal2=nn.BatchNorm1d(16)
        # self.query_proj = nn.Linear(query_dim, d_model)
        # self.key_proj = nn.Linear(key_value_dim, d_model)
        # self.value_proj = nn.Linear(key_value_dim, d_model)
        # self.encoder=nn.TransformerEncoderLayer(d_model=key_value_dim, nhead=4)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4,batch_first=True), num_layers=1,enable_nested_tensor=True)#不知道为啥这样写的效果远好于上面那行的
        # self.cls=torch.zeros(1,16,1)
        # Fixed positional encoding for B modality
        self.positional_encoding = FixedPositionalEncoding(d_model=d_model, max_len=num_tokens)
        # self.positional_encoding = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        
        # Attention layer
        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=16,vdim=d_model)
        # self.attention=nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        # self.gap=nn.AdaptiveAvgPool1d(1)
        # Output layer
        self.output_layer = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model*num_tokens, 1000), # Adjust output dimension as needed
            
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Linear(100,5),


            # nn.Linear(1997,100),
            # nn.ReLU(),
            # nn.Linear(16,5),
            # nn.Sigmoid()
        )

    def forward(self, A, B):
        
        # A=self.piclayernormal(A)
        A=torch.reshape(A,(A.shape[0],16,-1))
        # A=torch.cat([A]*16,dim=1)
        # B=self.speclayernormal(B)
        # B=torch.reshape(B,(B.shape[0],16*4,-1))
        
        # B=torch.reshape(B,(B.shape[0],16,-1))
        # cls=torch.zeros(A.shape[0],16,1).to("cuda:0")
        C=torch.cat([A,B],dim=2)
        C=C.transpose(1,2)
        C=self.positional_encoding(C)
        # C=C+self.positional_encoding
        # C=C.transpose(0,1)
        C=self.encoder(C)
        # C=C[:,0,:]

        # A is the Query, B is the Key and Value
        # query = self.query_proj(A)  # Shape: (batch_size, d_model)
        # B=self.layernormal(B) # Shape: (batch_size, channels, num_tokens)
        # # query=A
        # B = B.transpose(1, 2)  
        # B = self.positional_encoding(B)
        # # key = self.key_proj(B+self.positional_encoding) # Shape: (batch_size, num_tokens, d_model)
        # # value=B 
        # key = self.key_proj(B)
        # value = self.value_proj(B)  # Shape: (batch_size, num_tokens, d_model)

        # # Add positional encoding to key
        # # key = self.positional_encoding(key)

        # # Transpose for attention: (num_tokens, batch_size, d_model)
        # key = key.transpose(0, 1)
        # value = value.transpose(0, 1)
        # query = query.unsqueeze(0)  # Shape: (1, batch_size, d_model)

        # # Apply attention
        # attn_output, _ = self.attention(query, key, value)
        # # attn_output= self.attention(value,query)
        
        # # Transpose back: (batch_size, d_model)
        # attn_output = attn_output.squeeze(0)

        # C=C.transpose(1,2)
        # attn_output=self.gap(attn_output)
        # Output layer
        output = self.output_layer(C)
        return output