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
        self.layer_scale29 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)

    def forward(self, x344):
        x345=torch.permute(x344, [0, 3, 1, 2])
        x346=operator.mul(self.layer_scale29, x345)
        return x346

m = M().eval()
x344 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
