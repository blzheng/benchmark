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
        self.sigmoid5 = Sigmoid()

    def forward(self, x205):
        x206=self.sigmoid5(x205)
        return x206

m = M().eval()
x205 = torch.randn(torch.Size([1, 768, 1, 1]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)
