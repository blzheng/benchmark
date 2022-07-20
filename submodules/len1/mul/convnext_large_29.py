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

    def forward(self, x345):
        x346=operator.mul(self.layer_scale29, x345)
        return x346

m = M().eval()
x345 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x345)
end = time.time()
print(end-start)