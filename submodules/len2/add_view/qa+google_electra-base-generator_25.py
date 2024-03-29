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

    def forward(self, x291, x289):
        x292=operator.add(x291, (4, 64))
        x293=x289.view(x292)
        return x293

m = M().eval()
x291 = (1, 384, )
x289 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x291, x289)
end = time.time()
print(end-start)
