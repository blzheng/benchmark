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

    def forward(self, x67, x72):
        x73=x67.view(x72)
        return x73

m = M().eval()
x67 = torch.randn(torch.Size([1, 384, 768]))
x72 = (1, 384, 12, 64, )
start = time.time()
output = m(x67, x72)
end = time.time()
print(end-start)
