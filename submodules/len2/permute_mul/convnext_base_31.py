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
        self.layer_scale31 = torch.rand(torch.Size([512, 1, 1]))

    def forward(self, x366):
        x367=torch.permute(x366, [0, 3, 1, 2])
        x368=operator.mul(self.layer_scale31, x367)
        return x368

m = M().eval()
x366 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x366)
end = time.time()
print(end-start)
