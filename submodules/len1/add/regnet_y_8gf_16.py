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

    def forward(self, x267, x281):
        x282=operator.add(x267, x281)
        return x282

m = M().eval()
x267 = torch.randn(torch.Size([1, 2016, 7, 7]))
x281 = torch.randn(torch.Size([1, 2016, 7, 7]))
start = time.time()
output = m(x267, x281)
end = time.time()
print(end-start)
