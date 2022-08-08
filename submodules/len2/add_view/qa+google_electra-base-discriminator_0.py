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

    def forward(self, x32, x30):
        x33=operator.add(x32, (12, 64))
        x34=x30.view(x33)
        return x34

m = M().eval()
x32 = (1, 384, )
x30 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x32, x30)
end = time.time()
print(end-start)
