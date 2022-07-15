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
        self.relu118 = ReLU(inplace=True)

    def forward(self, x402):
        x403=self.relu118(x402)
        return x403

m = M().eval()
x402 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x402)
end = time.time()
print(end-start)
