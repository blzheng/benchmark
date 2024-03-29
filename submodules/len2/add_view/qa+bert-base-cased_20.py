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

    def forward(self, x242, x240):
        x243=operator.add(x242, (12, 64))
        x244=x240.view(x243)
        return x244

m = M().eval()
x242 = (1, 384, )
x240 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x242, x240)
end = time.time()
print(end-start)
