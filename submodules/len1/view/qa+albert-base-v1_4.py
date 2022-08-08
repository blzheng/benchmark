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

    def forward(self, x68, x77):
        x78=x68.view(x77)
        return x78

m = M().eval()
x68 = torch.randn(torch.Size([1, 384, 768]))
x77 = (1, 384, 12, 64, )
start = time.time()
output = m(x68, x77)
end = time.time()
print(end-start)
