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

    def forward(self, x774, x779):
        x780=operator.mul(x774, x779)
        return x780

m = M().eval()
x774 = torch.randn(torch.Size([1, 3840, 7, 7]))
x779 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x774, x779)
end = time.time()
print(end-start)
