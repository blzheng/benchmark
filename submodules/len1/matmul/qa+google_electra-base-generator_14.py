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

    def forward(self, x341, x342):
        x343=torch.matmul(x341, x342)
        return x343

m = M().eval()
x341 = torch.randn(torch.Size([1, 4, 384, 64]))
x342 = torch.randn(torch.Size([1, 4, 64, 384]))
start = time.time()
output = m(x341, x342)
end = time.time()
print(end-start)
