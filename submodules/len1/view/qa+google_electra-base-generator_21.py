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

    def forward(self, x247, x250):
        x251=x247.view(x250)
        return x251

m = M().eval()
x247 = torch.randn(torch.Size([1, 384, 256]))
x250 = (1, 384, 4, 64, )
start = time.time()
output = m(x247, x250)
end = time.time()
print(end-start)
