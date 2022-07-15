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

    def forward(self, x458, x450):
        x459=operator.add(x458, x450)
        return x459

m = M().eval()
x458 = torch.randn(torch.Size([1, 1024, 14, 14]))
x450 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x458, x450)
end = time.time()
print(end-start)
