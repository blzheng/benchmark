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

    def forward(self, x180, x193):
        x194=x180.view(x193)
        return x194

m = M().eval()
x180 = torch.randn(torch.Size([1, 384, 768]))
x193 = (1, 384, 12, 64, )
start = time.time()
output = m(x180, x193)
end = time.time()
print(end-start)
