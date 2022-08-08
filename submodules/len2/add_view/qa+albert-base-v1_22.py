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

    def forward(self, x298, x290):
        x299=operator.add(x298, (12, 64))
        x300=x290.view(x299)
        return x300

m = M().eval()
x298 = (1, 384, )
x290 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x298, x290)
end = time.time()
print(end-start)
