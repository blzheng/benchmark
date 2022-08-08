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

    def forward(self, x507):
        x508=x507.permute(0, 2, 1, 3)
        return x508

m = M().eval()
x507 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x507)
end = time.time()
print(end-start)
