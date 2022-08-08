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

    def forward(self, x368, x366):
        x369=operator.add(x368, (12, 64))
        x370=x366.view(x369)
        return x370

m = M().eval()
x368 = (1, 384, )
x366 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x368, x366)
end = time.time()
print(end-start)
