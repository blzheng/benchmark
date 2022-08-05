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
        self.dropout1 = Dropout(p=0.0, inplace=False)

    def forward(self, x23):
        x24=self.dropout1(x23)
        return x24

m = M().eval()
x23 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
