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

    def forward(self, x241, x249):
        x250=operator.add(x241, x249)
        x251=self.relu72(x250)
        return x251

m = M().eval()
x241 = torch.randn(torch.Size([1, 1008, 7, 7]))
x249 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x241, x249)
end = time.time()
print(end-start)
