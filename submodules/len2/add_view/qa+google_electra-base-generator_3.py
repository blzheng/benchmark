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

    def forward(self, x59, x57):
        x60=operator.add(x59, (256,))
        x61=x57.view(x60)
        return x61

m = M().eval()
x59 = (1, 384, )
x57 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x59, x57)
end = time.time()
print(end-start)
