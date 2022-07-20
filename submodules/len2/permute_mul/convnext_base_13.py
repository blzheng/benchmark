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
        self.layer_scale13 = torch.rand(torch.Size([512, 1, 1]))

    def forward(self, x168):
        x169=torch.permute(x168, [0, 3, 1, 2])
        x170=operator.mul(self.layer_scale13, x169)
        return x170

m = M().eval()
x168 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x168)
end = time.time()
print(end-start)
