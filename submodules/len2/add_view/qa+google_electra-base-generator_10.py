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

    def forward(self, x128, x114):
        x129=operator.add(x128, (4, 64))
        x130=x114.view(x129)
        return x130

m = M().eval()
x128 = (1, 384, )
x114 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x128, x114)
end = time.time()
print(end-start)
