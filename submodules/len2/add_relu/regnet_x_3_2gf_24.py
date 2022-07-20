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
        self.relu75 = ReLU(inplace=True)

    def forward(self, x251, x259):
        x260=operator.add(x251, x259)
        x261=self.relu75(x260)
        return x261

m = M().eval()
x251 = torch.randn(torch.Size([1, 1008, 7, 7]))
x259 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x251, x259)
end = time.time()
print(end-start)
