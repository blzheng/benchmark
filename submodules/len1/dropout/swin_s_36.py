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
        self.dropout36 = Dropout(p=0.0, inplace=False)

    def forward(self, x451):
        x452=self.dropout36(x451)
        return x452

m = M().eval()
x451 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x451)
end = time.time()
print(end-start)
