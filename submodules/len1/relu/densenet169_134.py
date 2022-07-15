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
        self.relu134 = ReLU(inplace=True)

    def forward(self, x476):
        x477=self.relu134(x476)
        return x477

m = M().eval()
x476 = torch.randn(torch.Size([1, 1120, 7, 7]))
start = time.time()
output = m(x476)
end = time.time()
print(end-start)
