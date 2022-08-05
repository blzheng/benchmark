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
        self.layer_scale20 = torch.rand(torch.Size([512, 1, 1])).to(torch.float32)

    def forward(self, x245):
        x246=torch.permute(x245, [0, 3, 1, 2])
        x247=operator.mul(self.layer_scale20, x246)
        return x247

m = M().eval()
x245 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x245)
end = time.time()
print(end-start)
