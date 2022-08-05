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
        self.linear34 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x267, x270):
        x271=x267.view(x270)
        x272=self.linear34(x271)
        return x272

m = M().eval()
x267 = torch.randn(torch.Size([1, 384, 4, 64]))
x270 = (1, 384, 256, )
start = time.time()
output = m(x267, x270)
end = time.time()
print(end-start)
