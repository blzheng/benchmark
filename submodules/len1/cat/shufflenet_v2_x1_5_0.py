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

    def forward(self, x9, x17):
        x18=torch.cat((x9, x17),dim=1)
        return x18

m = M().eval()
x9 = torch.randn(torch.Size([1, 88, 28, 28]))
x17 = torch.randn(torch.Size([1, 88, 28, 28]))
start = time.time()
output = m(x9, x17)
end = time.time()
print(end-start)
