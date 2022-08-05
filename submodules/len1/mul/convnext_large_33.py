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
        self.layer_scale33 = torch.rand(torch.Size([1536, 1, 1])).to(torch.float32)

    def forward(self, x395):
        x396=operator.mul(self.layer_scale33, x395)
        return x396

m = M().eval()
x395 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x395)
end = time.time()
print(end-start)
