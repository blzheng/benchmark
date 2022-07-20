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

    def forward(self, x264, x266, x267):
        x272=x271.view(x264, -1, x266, x267)
        return x272

m = M().eval()
x264 = 1
x266 = 14
x267 = 14
start = time.time()
output = m(x264, x266, x267)
end = time.time()
print(end-start)