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

    def forward(self, x511, x507):
        x512=x511.sigmoid()
        x513=operator.mul(x507, x512)
        return x513

m = M().eval()
x511 = torch.randn(torch.Size([1, 2064, 1, 1]))
x507 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x511, x507)
end = time.time()
print(end-start)