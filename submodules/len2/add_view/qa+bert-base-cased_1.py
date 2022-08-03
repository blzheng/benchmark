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

    def forward(self, x38, x36):
        x39=operator.add(x38, (12, 64))
        x40=x36.view(x39)
        return x40

m = M().eval()
x38 = (1, 384, )
x36 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x38, x36)
end = time.time()
print(end-start)
