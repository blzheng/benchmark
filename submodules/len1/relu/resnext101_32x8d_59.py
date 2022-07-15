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
        self.relu58 = ReLU(inplace=True)

    def forward(self, x205):
        x206=self.relu58(x205)
        return x206

m = M().eval()
x205 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)
