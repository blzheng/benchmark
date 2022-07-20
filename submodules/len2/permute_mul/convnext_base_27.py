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
        self.layer_scale27 = torch.rand(torch.Size([512, 1, 1]))

    def forward(self, x322):
        x323=torch.permute(x322, [0, 3, 1, 2])
        x324=operator.mul(self.layer_scale27, x323)
        return x324

m = M().eval()
x322 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x322)
end = time.time()
print(end-start)