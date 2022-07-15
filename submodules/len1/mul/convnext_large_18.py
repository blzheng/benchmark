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

    def forward(self, x224):
        x225=operator.mul(self.layer_scale18, x224)
        return x225

m = M().eval()
x224 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
