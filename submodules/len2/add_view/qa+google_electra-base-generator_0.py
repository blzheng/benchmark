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

    def forward(self, x33, x31):
        x34=operator.add(x33, (4, 64))
        x35=x31.view(x34)
        return x35

m = M().eval()
x33 = (1, 384, )
x31 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x33, x31)
end = time.time()
print(end-start)
