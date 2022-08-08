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
        self.dropout34 = Dropout(p=0.1, inplace=False)

    def forward(self, x515, x504):
        x516=self.dropout34(x515)
        x517=torch.matmul(x516, x504)
        return x517

m = M().eval()
x515 = torch.randn(torch.Size([1, 4, 384, 384]))
x504 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x515, x504)
end = time.time()
print(end-start)
