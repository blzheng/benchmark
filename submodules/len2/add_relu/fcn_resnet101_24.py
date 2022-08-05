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
        self.relu73 = ReLU(inplace=True)

    def forward(self, x260, x252):
        x261=operator.add(x260, x252)
        x262=self.relu73(x261)
        return x262

m = M().eval()
x260 = torch.randn(torch.Size([1, 1024, 28, 28]))
x252 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x260, x252)
end = time.time()
print(end-start)
