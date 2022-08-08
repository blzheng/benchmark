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

    def forward(self, x360, x357):
        x361=operator.add(x360, x357)
        return x361

m = M().eval()
x360 = torch.randn(torch.Size([1, 384, 768]))
x357 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x360, x357)
end = time.time()
print(end-start)
