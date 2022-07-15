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
        self.relu4 = ReLU(inplace=True)

    def forward(self, x11):
        x12=self.relu4(x11)
        return x12

m = M().eval()
x11 = torch.randn(torch.Size([1, 256, 13, 13]))
start = time.time()
output = m(x11)
end = time.time()
print(end-start)
