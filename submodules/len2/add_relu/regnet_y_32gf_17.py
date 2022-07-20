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
        self.relu72 = ReLU(inplace=True)

    def forward(self, x281, x295):
        x296=operator.add(x281, x295)
        x297=self.relu72(x296)
        return x297

m = M().eval()
x281 = torch.randn(torch.Size([1, 1392, 14, 14]))
x295 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x281, x295)
end = time.time()
print(end-start)
