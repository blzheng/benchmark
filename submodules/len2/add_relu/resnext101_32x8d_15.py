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
        self.relu46 = ReLU(inplace=True)

    def forward(self, x168, x160):
        x169=operator.add(x168, x160)
        x170=self.relu46(x169)
        return x170

m = M().eval()
x168 = torch.randn(torch.Size([1, 1024, 14, 14]))
x160 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x168, x160)
end = time.time()
print(end-start)
