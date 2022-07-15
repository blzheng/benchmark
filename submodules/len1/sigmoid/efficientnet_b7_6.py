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
        self.sigmoid6 = Sigmoid()

    def forward(self, x93):
        x94=self.sigmoid6(x93)
        return x94

m = M().eval()
x93 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x93)
end = time.time()
print(end-start)
