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

    def forward(self, x39, x31):
        x40=operator.add(x39, (12, 64))
        x41=x31.view(x40)
        return x41

m = M().eval()
x39 = (1, 384, )
x31 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x39, x31)
end = time.time()
print(end-start)
