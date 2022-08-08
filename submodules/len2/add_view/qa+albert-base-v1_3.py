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

    def forward(self, x71, x67):
        x72=operator.add(x71, (12, 64))
        x73=x67.view(x72)
        return x73

m = M().eval()
x71 = (1, 384, )
x67 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x71, x67)
end = time.time()
print(end-start)
