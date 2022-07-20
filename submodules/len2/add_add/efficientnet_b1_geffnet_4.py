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

    def forward(self, x292, x278, x307):
        x293=operator.add(x292, x278)
        x308=operator.add(x307, x293)
        return x308

m = M().eval()
x292 = torch.randn(torch.Size([1, 192, 7, 7]))
x278 = torch.randn(torch.Size([1, 192, 7, 7]))
x307 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x292, x278, x307)
end = time.time()
print(end-start)
