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

    def forward(self, x352, x337, x368):
        x353=operator.add(x352, x337)
        x369=operator.add(x368, x353)
        return x369

m = M().eval()
x352 = torch.randn(torch.Size([1, 232, 7, 7]))
x337 = torch.randn(torch.Size([1, 232, 7, 7]))
x368 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x352, x337, x368)
end = time.time()
print(end-start)
