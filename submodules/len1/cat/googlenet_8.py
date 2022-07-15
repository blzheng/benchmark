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

    def forward(self, x169, x175, x181, x185):
        x186=torch.cat([x169, x175, x181, x185], 1)
        return x186

m = M().eval()
x169 = torch.randn(torch.Size([1, 256, 7, 7]))
x175 = torch.randn(torch.Size([1, 320, 7, 7]))
x181 = torch.randn(torch.Size([1, 128, 7, 7]))
x185 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x169, x175, x181, x185)
end = time.time()
print(end-start)
