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
        self.relu40 = ReLU(inplace=True)

    def forward(self, x148, x140):
        x149=operator.add(x148, x140)
        x150=self.relu40(x149)
        return x150

m = M().eval()
x148 = torch.randn(torch.Size([1, 1024, 14, 14]))
x140 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x148, x140)
end = time.time()
print(end-start)
