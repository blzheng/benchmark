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

    def forward(self, x127, x113):
        x128=operator.add(x127, (12, 64))
        x129=x113.view(x128)
        return x129

m = M().eval()
x127 = (1, 384, )
x113 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x127, x113)
end = time.time()
print(end-start)
