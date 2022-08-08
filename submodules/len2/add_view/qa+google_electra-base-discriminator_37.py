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

    def forward(self, x416, x414):
        x417=operator.add(x416, (12, 64))
        x418=x414.view(x417)
        return x418

m = M().eval()
x416 = (1, 384, )
x414 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x416, x414)
end = time.time()
print(end-start)
