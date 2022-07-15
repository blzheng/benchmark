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
        self.sigmoid9 = Sigmoid()

    def forward(self, x269):
        x270=self.sigmoid9(x269)
        return x270

m = M().eval()
x269 = torch.randn(torch.Size([1, 768, 1, 1]))
start = time.time()
output = m(x269)
end = time.time()
print(end-start)
