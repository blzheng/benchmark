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
        self.linear40 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x309, x312):
        x313=x309.view(x312)
        x314=self.linear40(x313)
        return x314

m = M().eval()
x309 = torch.randn(torch.Size([1, 384, 4, 64]))
x312 = (1, 384, 256, )
start = time.time()
output = m(x309, x312)
end = time.time()
print(end-start)
