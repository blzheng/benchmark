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

    def forward(self, x361, x354, x356, x357):
        x362=x361.view(x354, -1, x356, x357)
        return x362

m = M().eval()
x361 = torch.randn(torch.Size([1, 488, 2, 7, 7]))
x354 = 1
x356 = 7
x357 = 7
start = time.time()
output = m(x361, x354, x356, x357)
end = time.time()
print(end-start)
