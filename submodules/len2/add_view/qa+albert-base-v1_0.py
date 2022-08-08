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

    def forward(self, x34, x30):
        x35=operator.add(x34, (12, 64))
        x36=x30.view(x35)
        return x36

m = M().eval()
x34 = (1, 384, )
x30 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x34, x30)
end = time.time()
print(end-start)
