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

    def forward(self, x507, x512):
        x513=operator.mul(x507, x512)
        return x513

m = M().eval()
x507 = torch.randn(torch.Size([1, 2064, 7, 7]))
x512 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x507, x512)
end = time.time()
print(end-start)
