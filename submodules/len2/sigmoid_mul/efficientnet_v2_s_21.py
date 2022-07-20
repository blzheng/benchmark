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
        self.sigmoid21 = Sigmoid()

    def forward(self, x405, x401):
        x406=self.sigmoid21(x405)
        x407=operator.mul(x406, x401)
        return x407

m = M().eval()
x405 = torch.randn(torch.Size([1, 1536, 1, 1]))
x401 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x405, x401)
end = time.time()
print(end-start)