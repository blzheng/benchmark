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

    def forward(self, x205, x208):
        x209=x205.view(x208)
        return x209

m = M().eval()
x205 = torch.randn(torch.Size([1, 384, 256]))
x208 = (1, 384, 4, 64, )
start = time.time()
output = m(x205, x208)
end = time.time()
print(end-start)
