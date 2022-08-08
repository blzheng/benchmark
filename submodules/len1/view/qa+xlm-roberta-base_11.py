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

    def forward(self, x140, x143):
        x144=x140.view(x143)
        return x144

m = M().eval()
x140 = torch.randn(torch.Size([1, 384, 12, 64]))
x143 = (1, 384, 768, )
start = time.time()
output = m(x140, x143)
end = time.time()
print(end-start)
