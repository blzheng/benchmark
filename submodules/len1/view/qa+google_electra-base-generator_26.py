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

    def forward(self, x282, x297):
        x298=x282.view(x297)
        return x298

m = M().eval()
x282 = torch.randn(torch.Size([1, 384, 256]))
x297 = (1, 384, 4, 64, )
start = time.time()
output = m(x282, x297)
end = time.time()
print(end-start)
