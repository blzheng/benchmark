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

    def forward(self, x328, x341):
        x342=x328.view(x341)
        return x342

m = M().eval()
x328 = torch.randn(torch.Size([1, 384, 768]))
x341 = (1, 384, 12, 64, )
start = time.time()
output = m(x328, x341)
end = time.time()
print(end-start)
