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

    def forward(self, x79, x82):
        x83=x79.view(x82)
        return x83

m = M().eval()
x79 = torch.randn(torch.Size([1, 384, 256]))
x82 = (1, 384, 4, 64, )
start = time.time()
output = m(x79, x82)
end = time.time()
print(end-start)
