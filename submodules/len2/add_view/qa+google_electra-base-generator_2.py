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

    def forward(self, x44, x30):
        x45=operator.add(x44, (4, 64))
        x46=x30.view(x45)
        return x46

m = M().eval()
x44 = (1, 384, )
x30 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x44, x30)
end = time.time()
print(end-start)
