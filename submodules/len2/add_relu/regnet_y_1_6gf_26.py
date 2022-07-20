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
        self.relu108 = ReLU(inplace=True)

    def forward(self, x427, x441):
        x442=operator.add(x427, x441)
        x443=self.relu108(x442)
        return x443

m = M().eval()
x427 = torch.randn(torch.Size([1, 888, 7, 7]))
x441 = torch.randn(torch.Size([1, 888, 7, 7]))
start = time.time()
output = m(x427, x441)
end = time.time()
print(end-start)
