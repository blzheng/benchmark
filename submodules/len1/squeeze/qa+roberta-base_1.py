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

    def forward(self, x536):
        x539=x536.squeeze(-1)
        return x539

m = M().eval()
x536 = torch.randn(torch.Size([1, 384, 1]))
start = time.time()
output = m(x536)
end = time.time()
print(end-start)
