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
        self.linear39 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout20 = Dropout(p=0.1, inplace=False)

    def forward(self, x308, x311, x280):
        x312=x308.view(x311)
        x313=self.linear39(x312)
        x314=self.dropout20(x313)
        x315=operator.add(x314, x280)
        return x315

m = M().eval()
x308 = torch.randn(torch.Size([1, 384, 12, 64]))
x311 = (1, 384, 768, )
x280 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x308, x311, x280)
end = time.time()
print(end-start)
