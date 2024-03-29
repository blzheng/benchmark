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

    def forward(self, x86, x72):
        x87=operator.add(x86, (4, 64))
        x88=x72.view(x87)
        return x88

m = M().eval()
x86 = (1, 384, )
x72 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x86, x72)
end = time.time()
print(end-start)
