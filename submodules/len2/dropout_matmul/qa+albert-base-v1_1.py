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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x90, x84):
        x91=self.dropout1(x90)
        x92=torch.matmul(x91, x84)
        return x92

m = M().eval()
x90 = torch.randn(torch.Size([1, 12, 384, 384]))
x84 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x90, x84)
end = time.time()
print(end-start)
