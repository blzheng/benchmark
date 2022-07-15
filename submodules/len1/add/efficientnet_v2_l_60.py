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

    def forward(self, x878, x863):
        x879=operator.add(x878, x863)
        return x879

m = M().eval()
x878 = torch.randn(torch.Size([1, 384, 7, 7]))
x863 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x878, x863)
end = time.time()
print(end-start)
