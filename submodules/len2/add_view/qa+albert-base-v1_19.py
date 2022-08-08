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

    def forward(self, x261, x253):
        x262=operator.add(x261, (12, 64))
        x263=x253.view(x262)
        return x263

m = M().eval()
x261 = (1, 384, )
x253 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x261, x253)
end = time.time()
print(end-start)
