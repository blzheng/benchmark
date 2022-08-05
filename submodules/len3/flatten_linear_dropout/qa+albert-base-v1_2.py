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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x130):
        x131=x130.flatten(2)
        x132=self.linear4(x131)
        x133=self.dropout2(x132)
        return x133

m = M().eval()
x130 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x130)
end = time.time()
print(end-start)
