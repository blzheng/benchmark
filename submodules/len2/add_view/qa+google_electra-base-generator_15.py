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

    def forward(self, x185, x183):
        x186=operator.add(x185, (256,))
        x187=x183.view(x186)
        return x187

m = M().eval()
x185 = (1, 384, )
x183 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x185, x183)
end = time.time()
print(end-start)
