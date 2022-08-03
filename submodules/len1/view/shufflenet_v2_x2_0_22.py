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

    def forward(self, x262, x264, x268, x266, x267):
        x269=x262.view(x264, 2, x268, x266, x267)
        return x269

m = M().eval()
x262 = torch.randn(torch.Size([1, 488, 14, 14]))
x264 = 1
x268 = 244
x266 = 14
x267 = 14
start = time.time()
output = m(x262, x264, x268, x266, x267)
end = time.time()
print(end-start)
