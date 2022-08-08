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

    def forward(self, x248, x246):
        x249=operator.add(x248, (12, 64))
        x250=x246.view(x249)
        return x250

m = M().eval()
x248 = (1, 384, )
x246 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x248, x246)
end = time.time()
print(end-start)
