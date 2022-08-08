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

    def forward(self, x285, x283):
        x286=operator.add(x285, (4, 64))
        x287=x283.view(x286)
        return x287

m = M().eval()
x285 = (1, 384, )
x283 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x285, x283)
end = time.time()
print(end-start)
