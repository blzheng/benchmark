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

    def forward(self, x605, x591, x620):
        x606=operator.add(x605, x591)
        x621=operator.add(x620, x606)
        return x621

m = M().eval()
x605 = torch.randn(torch.Size([1, 344, 7, 7]))
x591 = torch.randn(torch.Size([1, 344, 7, 7]))
x620 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x605, x591, x620)
end = time.time()
print(end-start)
