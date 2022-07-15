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
        self.relu88 = ReLU(inplace=True)

    def forward(self, x360):
        x361=self.relu88(x360)
        return x361

m = M().eval()
x360 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x360)
end = time.time()
print(end-start)
