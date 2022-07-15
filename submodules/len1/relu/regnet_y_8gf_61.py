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
        self.relu61 = ReLU(inplace=True)

    def forward(self, x251):
        x252=self.relu61(x251)
        return x252

m = M().eval()
x251 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x251)
end = time.time()
print(end-start)
