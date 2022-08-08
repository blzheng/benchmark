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
        self.dropout17 = Dropout(p=0.1, inplace=False)

    def forward(self, x272, x239):
        x273=self.dropout17(x272)
        x274=operator.add(x273, x239)
        return x274

m = M().eval()
x272 = torch.randn(torch.Size([1, 384, 256]))
x239 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x272, x239)
end = time.time()
print(end-start)
