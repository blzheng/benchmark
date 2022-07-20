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

    def forward(self, x389):
        x390=torch.permute(x389, [0, 2, 3, 1])
        return x390

m = M().eval()
x389 = torch.randn(torch.Size([1, 768, 7, 7]))
start = time.time()
output = m(x389)
end = time.time()
print(end-start)
