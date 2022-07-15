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

    def forward(self, x247, x259, x260):
        x261=torch.cat([x247, x259, x260], 1)
        return x261

m = M().eval()
x247 = torch.randn(torch.Size([1, 320, 5, 5]))
x259 = torch.randn(torch.Size([1, 192, 5, 5]))
x260 = torch.randn(torch.Size([1, 768, 5, 5]))
start = time.time()
output = m(x247, x259, x260)
end = time.time()
print(end-start)
