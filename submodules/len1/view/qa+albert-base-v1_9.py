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

    def forward(self, x141, x146):
        x147=x141.view(x146)
        return x147

m = M().eval()
x141 = torch.randn(torch.Size([1, 384, 768]))
x146 = (1, 384, 12, 64, )
start = time.time()
output = m(x141, x146)
end = time.time()
print(end-start)
