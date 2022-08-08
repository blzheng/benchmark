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

    def forward(self, x353, x351):
        x354=operator.add(x353, (256,))
        x355=x351.view(x354)
        return x355

m = M().eval()
x353 = (1, 384, )
x351 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x353, x351)
end = time.time()
print(end-start)
