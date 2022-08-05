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

    def forward(self, x517, x524):
        x525=operator.add(x517, x524)
        return x525

m = M().eval()
x517 = torch.randn(torch.Size([1, 14, 14, 384]))
x524 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x517, x524)
end = time.time()
print(end-start)
