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

    def forward(self, x30, x33):
        x34=x30.view(x33)
        return x34

m = M().eval()
x30 = torch.randn(torch.Size([1, 384, 768]))
x33 = (1, 384, 12, 64, )
start = time.time()
output = m(x30, x33)
end = time.time()
print(end-start)
