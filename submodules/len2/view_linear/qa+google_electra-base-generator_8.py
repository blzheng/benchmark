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
        self.linear52 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x393, x396):
        x397=x393.view(x396)
        x398=self.linear52(x397)
        return x398

m = M().eval()
x393 = torch.randn(torch.Size([1, 384, 4, 64]))
x396 = (1, 384, 256, )
start = time.time()
output = m(x393, x396)
end = time.time()
print(end-start)
