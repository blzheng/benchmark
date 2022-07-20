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
        self.dropout0 = Dropout(p=0.2, inplace=False)

    def forward(self, x150):
        x151=torch.flatten(x150, 1)
        x152=self.dropout0(x151)
        return x152

m = M().eval()
x150 = torch.randn(torch.Size([1, 1280, 1, 1]))
start = time.time()
output = m(x150)
end = time.time()
print(end-start)
