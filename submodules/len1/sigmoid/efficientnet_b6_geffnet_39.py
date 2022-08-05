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

    def forward(self, x586):
        x587=x586.sigmoid()
        return x587

m = M().eval()
x586 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x586)
end = time.time()
print(end-start)
