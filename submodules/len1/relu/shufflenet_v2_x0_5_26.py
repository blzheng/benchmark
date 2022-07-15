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
        self.relu26 = ReLU(inplace=True)

    def forward(self, x260):
        x261=self.relu26(x260)
        return x261

m = M().eval()
x260 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x260)
end = time.time()
print(end-start)
