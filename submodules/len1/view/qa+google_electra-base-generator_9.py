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

    def forward(self, x121, x124):
        x125=x121.view(x124)
        return x125

m = M().eval()
x121 = torch.randn(torch.Size([1, 384, 256]))
x124 = (1, 384, 4, 64, )
start = time.time()
output = m(x121, x124)
end = time.time()
print(end-start)
