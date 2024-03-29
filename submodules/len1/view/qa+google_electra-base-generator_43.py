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

    def forward(self, x477, x480):
        x481=x477.view(x480)
        return x481

m = M().eval()
x477 = torch.randn(torch.Size([1, 384, 4, 64]))
x480 = (1, 384, 256, )
start = time.time()
output = m(x477, x480)
end = time.time()
print(end-start)
