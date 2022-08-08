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

    def forward(self, x251, x281):
        x282=operator.add(x251, x281)
        return x282

m = M().eval()
x251 = torch.randn(torch.Size([1, 384, 768]))
x281 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x251, x281)
end = time.time()
print(end-start)
