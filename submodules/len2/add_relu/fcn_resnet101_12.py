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
        self.relu37 = ReLU(inplace=True)

    def forward(self, x140, x132):
        x141=operator.add(x140, x132)
        x142=self.relu37(x141)
        return x142

m = M().eval()
x140 = torch.randn(torch.Size([1, 1024, 28, 28]))
x132 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x140, x132)
end = time.time()
print(end-start)
