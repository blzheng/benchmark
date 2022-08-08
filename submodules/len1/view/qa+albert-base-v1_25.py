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

    def forward(self, x327, x336):
        x337=x327.view(x336)
        return x337

m = M().eval()
x327 = torch.randn(torch.Size([1, 384, 768]))
x336 = (1, 384, 12, 64, )
start = time.time()
output = m(x327, x336)
end = time.time()
print(end-start)
