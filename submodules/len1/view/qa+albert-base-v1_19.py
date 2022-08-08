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

    def forward(self, x253, x262):
        x263=x253.view(x262)
        return x263

m = M().eval()
x253 = torch.randn(torch.Size([1, 384, 768]))
x262 = (1, 384, 12, 64, )
start = time.time()
output = m(x253, x262)
end = time.time()
print(end-start)
