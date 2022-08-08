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

    def forward(self, x450, x465):
        x466=x450.view(x465)
        return x466

m = M().eval()
x450 = torch.randn(torch.Size([1, 384, 256]))
x465 = (1, 384, 4, 64, )
start = time.time()
output = m(x450, x465)
end = time.time()
print(end-start)
