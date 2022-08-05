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
        self.layer_scale35 = torch.rand(torch.Size([1024, 1, 1])).to(torch.float32)

    def forward(self, x416):
        x417=torch.permute(x416, [0, 3, 1, 2])
        x418=operator.mul(self.layer_scale35, x417)
        return x418

m = M().eval()
x416 = torch.randn(torch.Size([1, 7, 7, 1024]))
start = time.time()
output = m(x416)
end = time.time()
print(end-start)
