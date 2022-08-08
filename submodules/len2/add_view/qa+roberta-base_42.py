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

    def forward(self, x463, x449):
        x464=operator.add(x463, (12, 64))
        x465=x449.view(x464)
        return x465

m = M().eval()
x463 = (1, 384, )
x449 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x463, x449)
end = time.time()
print(end-start)
