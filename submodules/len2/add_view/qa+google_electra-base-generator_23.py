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

    def forward(self, x269, x267):
        x270=operator.add(x269, (256,))
        x271=x267.view(x270)
        return x271

m = M().eval()
x269 = (1, 384, )
x267 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x269, x267)
end = time.time()
print(end-start)
