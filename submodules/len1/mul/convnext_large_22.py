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

    def forward(self, x268):
        x269=operator.mul(self.layer_scale22, x268)
        return x269

m = M().eval()
x268 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x268)
end = time.time()
print(end-start)