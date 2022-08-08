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

    def forward(self, x114, x129):
        x130=x114.view(x129)
        return x130

m = M().eval()
x114 = torch.randn(torch.Size([1, 384, 256]))
x129 = (1, 384, 4, 64, )
start = time.time()
output = m(x114, x129)
end = time.time()
print(end-start)
