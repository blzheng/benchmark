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
        self.relu109 = ReLU(inplace=True)

    def forward(self, x372):
        x373=self.relu109(x372)
        return x373

m = M().eval()
x372 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x372)
end = time.time()
print(end-start)
