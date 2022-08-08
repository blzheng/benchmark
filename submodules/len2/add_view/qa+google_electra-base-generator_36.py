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

    def forward(self, x411, x409):
        x412=operator.add(x411, (4, 64))
        x413=x409.view(x412)
        return x413

m = M().eval()
x411 = (1, 384, )
x409 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x411, x409)
end = time.time()
print(end-start)
