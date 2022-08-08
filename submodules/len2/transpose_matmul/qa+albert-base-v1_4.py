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

    def forward(self, x190, x185):
        x196=x190.transpose(-1, -2)
        x197=torch.matmul(x185, x196)
        return x197

m = M().eval()
x190 = torch.randn(torch.Size([1, 12, 384, 64]))
x185 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x190, x185)
end = time.time()
print(end-start)
