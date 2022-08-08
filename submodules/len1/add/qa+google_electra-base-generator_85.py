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

    def forward(self, x531, x527):
        x532=operator.add(x531, x527)
        return x532

m = M().eval()
x531 = torch.randn(torch.Size([1, 384, 256]))
x527 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x531, x527)
end = time.time()
print(end-start)
