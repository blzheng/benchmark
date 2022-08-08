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

    def forward(self, x29, x44):
        x45=x29.view(x44)
        return x45

m = M().eval()
x29 = torch.randn(torch.Size([1, 384, 768]))
x44 = (1, 384, 12, 64, )
start = time.time()
output = m(x29, x44)
end = time.time()
print(end-start)
