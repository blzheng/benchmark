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

    def forward(self, x362, x392):
        x393=operator.add(x362, x392)
        return x393

m = M().eval()
x362 = torch.randn(torch.Size([1, 384, 768]))
x392 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x362, x392)
end = time.time()
print(end-start)