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

    def forward(self, x498, x501):
        x502=x498.view(x501)
        return x502

m = M().eval()
x498 = torch.randn(torch.Size([1, 384, 768]))
x501 = (1, 384, 12, 64, )
start = time.time()
output = m(x498, x501)
end = time.time()
print(end-start)
