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
        self.layer_scale12 = torch.rand(torch.Size([768, 1, 1]))

    def forward(self, x157):
        x158=torch.permute(x157, [0, 3, 1, 2])
        x159=operator.mul(self.layer_scale12, x158)
        return x159

m = M().eval()
x157 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x157)
end = time.time()
print(end-start)
