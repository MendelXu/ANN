from torch import nn




class NullBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x
