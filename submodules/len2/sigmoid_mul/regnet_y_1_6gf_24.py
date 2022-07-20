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
        self.sigmoid24 = Sigmoid()

    def forward(self, x403, x399):
        x404=self.sigmoid24(x403)
        x405=operator.mul(x404, x399)
        return x405

m = M().eval()
x403 = torch.randn(torch.Size([1, 336, 1, 1]))
x399 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x403, x399)
end = time.time()
print(end-start)
