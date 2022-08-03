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

    def forward(self, x113, x128):
        x129=x113.view(x128)
        return x129

m = M().eval()
x113 = torch.randn(torch.Size([1, 384, 768]))
x128 = (1, 384, 12, 64, )
start = time.time()
output = m(x113, x128)
end = time.time()
print(end-start)
