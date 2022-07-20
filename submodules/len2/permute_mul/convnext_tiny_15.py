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
        self.layer_scale15 = torch.rand(torch.Size([768, 1, 1]))

    def forward(self, x196):
        x197=torch.permute(x196, [0, 3, 1, 2])
        x198=operator.mul(self.layer_scale15, x197)
        return x198

m = M().eval()
x196 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x196)
end = time.time()
print(end-start)
