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

    def forward(self, x354, x358, x356, x357):
        x359=x352.view(x354, 2, x358, x356, x357)
        return x359

m = M().eval()
x354 = 1
x358 = 96
x356 = 7
x357 = 7
start = time.time()
output = m(x354, x358, x356, x357)
end = time.time()
print(end-start)
