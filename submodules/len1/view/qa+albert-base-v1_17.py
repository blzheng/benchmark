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

    def forward(self, x217, x230):
        x231=x217.view(x230)
        return x231

m = M().eval()
x217 = torch.randn(torch.Size([1, 384, 768]))
x230 = (1, 384, 12, 64, )
start = time.time()
output = m(x217, x230)
end = time.time()
print(end-start)
