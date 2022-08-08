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

    def forward(self, x333, x331):
        x334=operator.add(x333, (4, 64))
        x335=x331.view(x334)
        return x335

m = M().eval()
x333 = (1, 384, )
x331 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x333, x331)
end = time.time()
print(end-start)
