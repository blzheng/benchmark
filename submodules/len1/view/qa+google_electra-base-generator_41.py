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

    def forward(self, x457, x460):
        x461=x457.view(x460)
        return x461

m = M().eval()
x457 = torch.randn(torch.Size([1, 384, 256]))
x460 = (1, 384, 4, 64, )
start = time.time()
output = m(x457, x460)
end = time.time()
print(end-start)
