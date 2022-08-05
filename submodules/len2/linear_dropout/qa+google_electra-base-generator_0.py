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
        self.linear4 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x61):
        x62=self.linear4(x61)
        x63=self.dropout2(x62)
        return x63

m = M().eval()
x61 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
