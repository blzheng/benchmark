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
        self.layer_scale34 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)

    def forward(self, x405):
        x406=torch.permute(x405, [0, 3, 1, 2])
        x407=operator.mul(self.layer_scale34, x406)
        return x407

m = M().eval()
x405 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x405)
end = time.time()
print(end-start)
