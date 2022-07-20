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

    def forward(self, x258, x254):
        x259=x258.sigmoid()
        x260=operator.mul(x254, x259)
        return x260

m = M().eval()
x258 = torch.randn(torch.Size([1, 864, 1, 1]))
x254 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x258, x254)
end = time.time()
print(end-start)
