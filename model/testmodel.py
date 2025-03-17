import torch
from torch import nn
from . import resnet
from .muti_scale_pc import ECA

class mlpmodel(nn.Module):

    def __init__(self, in_channel=1, out_channel=5):
        super(mlpmodel, self).__init__()
        self.fc = nn.Sequential(
            # [BatchSize, 2688]
            
            nn.Linear(3844, 10000),
            # [BatchSize, 100]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(10000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, out_channel)
            # [BatchSize, 2]
        )

    def forward(self, x):
        # [BatchSize, 2688]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # [BatchSize, 2]
        return x
    
class starnet(nn.Module): #https://academic.oup.com/mnras/article/475/3/2978/4775133#110346236

    def __init__(self, in_channel=1, out_channel=5):
        super(starnet, self).__init__()
        self.conv = nn.Sequential(
            # [BatchSize, 2688]
            nn.Conv1d(in_channels=in_channel, out_channels=4, kernel_size=(3,), stride=(1,), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=(3,), stride=(1,), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(5,), stride=(4,), padding=2)

            # [BatchSize, 2]
        )
        self.fc = nn.Sequential(
            # [BatchSize, 2688]
            
            nn.Linear(15376, 256),
            # [BatchSize, 100]
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(128, out_channel),
            # nn.Sigmoid(),
            # [BatchSize, 2]
        )


    def forward(self, x):
        # [BatchSize, 2688]
        x=self.conv(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # [BatchSize, 2]
        return x
class AFGK_star(nn.Module): #https://www.degruyter.com/document/doi/10.1515/astro-2022-0209/html

    def __init__(self, in_channel=1, out_channel=5):
        super(AFGK_star, self).__init__()
        self.conv = nn.Sequential(
            # [BatchSize, 2688]
            nn.Conv1d(in_channels=in_channel, out_channels=128, kernel_size=(7,), stride=(1,), padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(7,), stride=(1,), padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(7,), stride=(1,), padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # [BatchSize, 2]
        )
        self.fc = nn.Sequential(
            # [BatchSize, 2688]
            
            nn.Linear(492032, 1024),
            # [BatchSize, 100]
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, out_channel),
            # nn.Sigmoid(),
            # [BatchSize, 2]
        )


    def forward(self, x):
        # [BatchSize, 2688]
        x=self.conv(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # [BatchSize, 2]
        return x
class SPNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, with_class:bool =True):
        super(SPNet, self).__init__()
        channels=[8,256,16]
        self.layer1_1 = nn.Sequential(
            # [BatchSize, 1, 6284]
            nn.Conv1d(in_channels=in_channel, out_channels=channels[0], kernel_size=(8,), stride=(1,), padding=1,bias=False),
            # [BatchSize, 16, 783]
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,),padding=0)
            # [BatchSize, 16, 392]
        )


        self.layer2 = nn.Sequential(
            # [BatchSize, 16, 684]
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3,), stride=(1,), padding=0,bias=False),
            # [BatchSize, 32, 684]
            nn.BatchNorm1d(channels[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 32, 342]
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=(1,), stride=(1,), padding=0,bias=False),
            # [BatchSize, 64, 342]
            nn.BatchNorm1d(channels[2]),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 64, 171]
        )



        numfeatures=30672##1912*8#15296#30656


        #1600
        if with_class:
            self.fc = nn.Sequential(
                # [BatchSize, 2688]
                # nn.Dropout(p=0.5),
                # nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                # kan.KAN([numfeatures,100,out_channel]),
                
                # [BatchSize, 2]
                # nn.Dropout(p=0.5),
                nn.Linear(numfeatures, 1000),
                # # # [BatchSize, 100]
                nn.ReLU(),
                # nn.BatchNorm1d(1000),
                # nn.Dropout(p=0.5),
                nn.Linear(1000, 100),
                nn.ReLU(),
                # nn.BatchNorm1d(100),
                # nn.Dropout(p=0.5),
                nn.Linear(100, out_channel),
                # nn.Sigmoid()
            )
        else:
            self.fc=nn.Sequential()

    def forward(self, x):
        x1 = self.layer1_1(x)
#        x2 = self.layer1_2(x)
#        x3 = self.layer1_3(x)
#        x = torch.cat((x1，x2), dim=2)
        x = self.layer2(x1)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        # [BatchSize, 64, 42]
        # x = x.view(x.size(0), -1)
        # [BatchSize, 2688]
        x = self.fc(x)
        # [BatchSize, 2]
        return x
    
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
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
    def __init__(self, query_dim=int(512 * 2.5), key_value_dim=16, num_tokens=1917, d_model=16):
        super(MultiModalModel, self).__init__()
        
        # Linear layers to project A and B to the model dimension
        self.backbone=SPNet(with_class=False)
        self.speclayernormal=nn.LayerNorm([16,1917])
        # self.query_proj = nn.Linear(query_dim, d_model)
        # self.key_proj = nn.Linear(key_value_dim, d_model)
        # self.value_proj = nn.Linear(key_value_dim, d_model)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4,batch_first=True), num_layers=1,enable_nested_tensor=True)
        # Fixed positional encoding for B modality
        self.positional_encoding = FixedPositionalEncoding(key_value_dim, max_len=num_tokens)
        # self.eca=ECA(channel=1917)
        # self.positional_encoding = nn.Parameter(torch.zeros(1, num_tokens, key_value_dim))
        
        # Attention layer
        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=16,vdim=d_model)
        # self.attention=nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model*num_tokens, 1000) , # Adjust output dimension as needed
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Linear(100,5),
            # nn.Sigmoid()
        )

    def forward(self, x):
        B=self.backbone(x)
        # B=self.speclayernormal(B)
        # B=torch.reshape(B,(B.shape[0],16,-1))
        C=B.transpose(1,2)
        C=self.positional_encoding(C)
        # C=C.transpose(0,1)
        C=self.encoder(C)
        # C=self.eca(C)
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

        # attn_output=C.transpose(0,1)
        # Output layer
        output = self.output_layer(C)
        return output
    




class attresnet(nn.Module):

    def __init__(self, in_channel=1, out_channel=5,d_model=16,num_tokens=32,key_value_dim=16):
        super(attresnet, self).__init__()
        self.mob=resnet.resnet18(num_classes=5,include_top=False)
        # self.layernormal=nn.LayerNorm(1280)

        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4,batch_first=True), num_layers=1,enable_nested_tensor=True)
        # Fixed positional encoding for B modality
        self.positional_encoding = FixedPositionalEncoding(key_value_dim, max_len=num_tokens)
        
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Linear(100,5),
        )

    def forward(self, x):
        x=self.mob(x)
        # x=self.layernormal(x)
        x=torch.reshape(x,(x.shape[0],16,-1))
        # B=torch.reshape(B,(B.shape[0],16,-1))

        x=x.transpose(1,2)
        x=self.positional_encoding(x)
        # C=C.transpose(0,1)
        x=self.encoder(x)
        # x=self.eca(x)

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

        # attn_output=C.transpose(0,1)
        # Output layer
        x = self.output_layer(x)
        return x
class SPNet2(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, with_class:bool =True):
        super(SPNet2, self).__init__()
        channels=[16,256,16]
        self.layer1_1 = nn.Sequential(
            # [BatchSize, 1, 6284]
            nn.Conv1d(in_channels=in_channel, out_channels=channels[0], kernel_size=(8,), stride=(1,), padding=1,bias=False),
            # [BatchSize, 16, 783]
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,),padding=0)
            # [BatchSize, 16, 392]
        )


        self.layer2 = nn.Sequential(
            # [BatchSize, 16, 684]
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3,), stride=(1,), padding=1,bias=False),
            # [BatchSize, 32, 684]
            nn.BatchNorm1d(channels[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 32, 342]
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3,), stride=(1,), padding=1,bias=False),
            # [BatchSize, 64, 342]
            nn.BatchNorm1d(channels[2]),
            # nn.Dropout(p=0.5),
            
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 64, 171]
        )



        numfeatures=30704##1912*8#15296#30656


        #1600
        if with_class:
            self.fc = nn.Sequential(
                # [BatchSize, 2688]
                # nn.Dropout(p=0.5),
                # nn.AdaptiveMaxPool1d(1),
                nn.ReLU(),
                nn.Flatten(),
                # kan.KAN([numfeatures,100,out_channel]),
                
                # [BatchSize, 2]
                # nn.Dropout(p=0.5),
                nn.Linear(numfeatures, 1000),
                # # # [BatchSize, 100]
                nn.ReLU(),
                # nn.BatchNorm1d(1000),
                # nn.Dropout(p=0.5),
                nn.Linear(1000, 100),
                nn.ReLU(),
                # nn.BatchNorm1d(100),
                # nn.Dropout(p=0.5),
                nn.Linear(100, out_channel),
                # nn.Sigmoid()
            )
        else:
            self.fc=nn.Sequential()

    def forward(self, x):
        x1 = self.layer1_1(x)
#        x2 = self.layer1_2(x)
#        x3 = self.layer1_3(x)
#        x = torch.cat((x1，x2), dim=2)
        x = self.layer2(x1)
        x = self.layer3(x)+x1
        # x = self.layer4(x)
        # x = self.layer5(x)
        # [BatchSize, 64, 42]
        # x = x.view(x.size(0), -1)
        # [BatchSize, 2688]
        x = self.fc(x)
        # [BatchSize, 2]
        return x