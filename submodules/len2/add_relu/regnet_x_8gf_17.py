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
        self.relu54 = ReLU(inplace=True)

    def forward(self, x179, x187):
        x188=operator.add(x179, x187)
        x189=self.relu54(x188)
        return x189

m = M().eval()
x179 = torch.randn(torch.Size([1, 720, 14, 14]))
x187 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x179, x187)
end = time.time()
print(end-start)
