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

    def forward(self, x253, x248):
        x254=operator.mul(x253, x248)
        return x254

m = M().eval()
x253 = torch.randn(torch.Size([1, 864, 1, 1]))
x248 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x253, x248)
end = time.time()
print(end-start)
