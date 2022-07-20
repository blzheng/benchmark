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

    def forward(self, x304, x289, x320):
        x305=operator.add(x304, x289)
        x321=operator.add(x320, x305)
        return x321

m = M().eval()
x304 = torch.randn(torch.Size([1, 192, 7, 7]))
x289 = torch.randn(torch.Size([1, 192, 7, 7]))
x320 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x304, x289, x320)
end = time.time()
print(end-start)
