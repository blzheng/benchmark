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

    def forward(self, x293, x289):
        x294=operator.add(x293, (12, 64))
        x295=x289.view(x294)
        return x295

m = M().eval()
x293 = (1, 384, )
x289 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x293, x289)
end = time.time()
print(end-start)
