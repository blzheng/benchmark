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
        self.linear51 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout26 = Dropout(p=0.1, inplace=False)

    def forward(self, x392, x395):
        x396=x392.view(x395)
        x397=self.linear51(x396)
        x398=self.dropout26(x397)
        return x398

m = M().eval()
x392 = torch.randn(torch.Size([1, 384, 12, 64]))
x395 = (1, 384, 768, )
start = time.time()
output = m(x392, x395)
end = time.time()
print(end-start)
