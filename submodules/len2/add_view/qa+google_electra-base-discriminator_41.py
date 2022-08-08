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

    def forward(self, x458, x456):
        x459=operator.add(x458, (12, 64))
        x460=x456.view(x459)
        return x460

m = M().eval()
x458 = (1, 384, )
x456 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x458, x456)
end = time.time()
print(end-start)
