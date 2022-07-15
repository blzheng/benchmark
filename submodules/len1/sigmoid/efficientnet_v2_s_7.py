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
        self.sigmoid7 = Sigmoid()

    def forward(self, x183):
        x184=self.sigmoid7(x183)
        return x184

m = M().eval()
x183 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
