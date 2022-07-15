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

    def forward(self, x524, x519):
        x525=operator.mul(x524, x519)
        return x525

m = M().eval()
x524 = torch.randn(torch.Size([1, 1344, 1, 1]))
x519 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x524, x519)
end = time.time()
print(end-start)
