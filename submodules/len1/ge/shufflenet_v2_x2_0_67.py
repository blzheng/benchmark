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

    def forward(self, x263):
        x267=operator.getitem(x263, 3)
        return x267

m = M().eval()
x263 = (1, 488, 14, 14, )
start = time.time()
output = m(x263)
end = time.time()
print(end-start)
