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
        self.dropout19 = Dropout(p=0.1, inplace=False)

    def forward(self, x304, x293):
        x305=self.dropout19(x304)
        x306=torch.matmul(x305, x293)
        return x306

m = M().eval()
x304 = torch.randn(torch.Size([1, 12, 384, 384]))
x293 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x304, x293)
end = time.time()
print(end-start)
