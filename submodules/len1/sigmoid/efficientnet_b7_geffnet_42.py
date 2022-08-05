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

    def forward(self, x629):
        x630=x629.sigmoid()
        return x630

m = M().eval()
x629 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x629)
end = time.time()
print(end-start)
