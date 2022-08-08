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

    def forward(self, x246, x249):
        x250=x246.view(x249)
        return x250

m = M().eval()
x246 = torch.randn(torch.Size([1, 384, 768]))
x249 = (1, 384, 12, 64, )
start = time.time()
output = m(x246, x249)
end = time.time()
print(end-start)