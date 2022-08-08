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

    def forward(self, x459, x457):
        x460=operator.add(x459, (4, 64))
        x461=x457.view(x460)
        return x461

m = M().eval()
x459 = (1, 384, )
x457 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x459, x457)
end = time.time()
print(end-start)
