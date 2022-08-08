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

    def forward(self, x365, x380):
        x381=x365.view(x380)
        return x381

m = M().eval()
x365 = torch.randn(torch.Size([1, 384, 768]))
x380 = (1, 384, 12, 64, )
start = time.time()
output = m(x365, x380)
end = time.time()
print(end-start)
