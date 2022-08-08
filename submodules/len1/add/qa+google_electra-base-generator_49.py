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

    def forward(self, x315, x281):
        x316=operator.add(x315, x281)
        return x316

m = M().eval()
x315 = torch.randn(torch.Size([1, 384, 256]))
x281 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x315, x281)
end = time.time()
print(end-start)
