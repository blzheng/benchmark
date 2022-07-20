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
        self.layer_scale9 = torch.rand(torch.Size([512, 1, 1]))

    def forward(self, x124):
        x125=torch.permute(x124, [0, 3, 1, 2])
        x126=operator.mul(self.layer_scale9, x125)
        return x126

m = M().eval()
x124 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x124)
end = time.time()
print(end-start)
