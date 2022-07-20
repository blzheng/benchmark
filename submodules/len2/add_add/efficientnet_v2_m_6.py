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

    def forward(self, x763, x748, x779):
        x764=operator.add(x763, x748)
        x780=operator.add(x779, x764)
        return x780

m = M().eval()
x763 = torch.randn(torch.Size([1, 512, 7, 7]))
x748 = torch.randn(torch.Size([1, 512, 7, 7]))
x779 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x763, x748, x779)
end = time.time()
print(end-start)
