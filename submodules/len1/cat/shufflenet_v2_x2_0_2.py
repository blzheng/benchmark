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

    def forward(self, x52, x61):
        x62=torch.cat((x52, x61),dim=1)
        return x62

m = M().eval()
x52 = torch.randn(torch.Size([1, 122, 28, 28]))
x61 = torch.randn(torch.Size([1, 122, 28, 28]))
start = time.time()
output = m(x52, x61)
end = time.time()
print(end-start)
