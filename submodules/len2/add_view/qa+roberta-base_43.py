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

    def forward(self, x478, x476):
        x479=operator.add(x478, (768,))
        x480=x476.view(x479)
        return x480

m = M().eval()
x478 = (1, 384, )
x476 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x478, x476)
end = time.time()
print(end-start)
