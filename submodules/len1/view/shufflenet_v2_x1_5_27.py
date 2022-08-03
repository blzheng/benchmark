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

    def forward(self, x317, x310, x312, x313):
        x318=x317.view(x310, -1, x312, x313)
        return x318

m = M().eval()
x317 = torch.randn(torch.Size([1, 352, 2, 7, 7]))
x310 = 1
x312 = 7
x313 = 7
start = time.time()
output = m(x317, x310, x312, x313)
end = time.time()
print(end-start)
