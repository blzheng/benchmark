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

    def forward(self, x415, x418):
        x419=x415.view(x418)
        return x419

m = M().eval()
x415 = torch.randn(torch.Size([1, 384, 256]))
x418 = (1, 384, 4, 64, )
start = time.time()
output = m(x415, x418)
end = time.time()
print(end-start)
