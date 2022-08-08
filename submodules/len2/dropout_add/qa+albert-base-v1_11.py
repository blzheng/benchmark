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
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x465, x436):
        x466=self.dropout2(x465)
        x467=operator.add(x436, x466)
        return x467

m = M().eval()
x465 = torch.randn(torch.Size([1, 384, 768]))
x436 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x465, x436)
end = time.time()
print(end-start)
