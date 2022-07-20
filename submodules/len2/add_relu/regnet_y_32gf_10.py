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
        self.relu44 = ReLU(inplace=True)

    def forward(self, x169, x183):
        x184=operator.add(x169, x183)
        x185=self.relu44(x184)
        return x185

m = M().eval()
x169 = torch.randn(torch.Size([1, 1392, 14, 14]))
x183 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x169, x183)
end = time.time()
print(end-start)
