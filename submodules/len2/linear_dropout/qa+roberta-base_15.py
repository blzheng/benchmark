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
        self.linear47 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout24 = Dropout(p=0.1, inplace=False)

    def forward(self, x360):
        x361=self.linear47(x360)
        x362=self.dropout24(x361)
        return x362

m = M().eval()
x360 = torch.randn(torch.Size([1, 384, 3072]))
start = time.time()
output = m(x360)
end = time.time()
print(end-start)
