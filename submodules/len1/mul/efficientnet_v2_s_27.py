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

    def forward(self, x502, x497):
        x503=operator.mul(x502, x497)
        return x503

m = M().eval()
x502 = torch.randn(torch.Size([1, 1536, 1, 1]))
x497 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x502, x497)
end = time.time()
print(end-start)