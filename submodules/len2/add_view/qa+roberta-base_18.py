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

    def forward(self, x211, x197):
        x212=operator.add(x211, (12, 64))
        x213=x197.view(x212)
        return x213

m = M().eval()
x211 = (1, 384, )
x197 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x211, x197)
end = time.time()
print(end-start)
