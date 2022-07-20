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

    def forward(self, x316):
        x317=x316.contiguous()
        return x317

m = M().eval()
x316 = torch.randn(torch.Size([1, 352, 2, 7, 7]))
start = time.time()
output = m(x316)
end = time.time()
print(end-start)
