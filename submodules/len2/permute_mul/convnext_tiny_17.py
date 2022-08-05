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
        self.layer_scale17 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)

    def forward(self, x218):
        x219=torch.permute(x218, [0, 3, 1, 2])
        x220=operator.mul(self.layer_scale17, x219)
        return x220

m = M().eval()
x218 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
