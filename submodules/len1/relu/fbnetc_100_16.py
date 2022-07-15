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
        self.relu16 = ReLU(inplace=True)

    def forward(self, x76):
        x77=self.relu16(x76)
        return x77

m = M().eval()
x76 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x76)
end = time.time()
print(end-start)
