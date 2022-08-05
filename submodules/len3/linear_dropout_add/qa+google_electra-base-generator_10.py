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
        self.linear34 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout17 = Dropout(p=0.1, inplace=False)

    def forward(self, x271, x239):
        x272=self.linear34(x271)
        x273=self.dropout17(x272)
        x274=operator.add(x273, x239)
        return x274

m = M().eval()
x271 = torch.randn(torch.Size([1, 384, 256]))
x239 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x271, x239)
end = time.time()
print(end-start)
