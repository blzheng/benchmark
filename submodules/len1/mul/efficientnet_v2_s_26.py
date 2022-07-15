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

    def forward(self, x486, x481):
        x487=operator.mul(x486, x481)
        return x487

m = M().eval()
x486 = torch.randn(torch.Size([1, 1536, 1, 1]))
x481 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x486, x481)
end = time.time()
print(end-start)
