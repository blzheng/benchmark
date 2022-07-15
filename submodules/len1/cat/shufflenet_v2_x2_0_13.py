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

    def forward(self, x298, x307):
        x308=torch.cat((x298, x307),dim=1)
        return x308

m = M().eval()
x298 = torch.randn(torch.Size([1, 488, 7, 7]))
x307 = torch.randn(torch.Size([1, 488, 7, 7]))
start = time.time()
output = m(x298, x307)
end = time.time()
print(end-start)
