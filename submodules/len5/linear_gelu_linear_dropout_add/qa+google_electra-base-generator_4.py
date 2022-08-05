import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.linear29 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear30 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout15 = Dropout(p=0.1, inplace=False)

    def forward(self, x233, x233):
        x234=self.linear29(x233)
        x235=torch._C._nn.gelu(x234)
        x236=self.linear30(x235)
        x237=self.dropout15(x236)
        x238=operator.add(x237, x233)
        return x238

m = M().eval()
x233 = torch.randn(torch.Size([1, 384, 256]))
x233 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x233, x233)
end = time.time()
print(end-start)
