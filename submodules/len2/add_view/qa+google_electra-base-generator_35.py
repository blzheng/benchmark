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

    def forward(self, x395, x393):
        x396=operator.add(x395, (256,))
        x397=x393.view(x396)
        return x397

m = M().eval()
x395 = (1, 384, )
x393 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x395, x393)
end = time.time()
print(end-start)
