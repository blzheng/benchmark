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
        self.dropout35 = Dropout(p=0.1, inplace=False)

    def forward(self, x523, x490):
        x524=self.dropout35(x523)
        x525=operator.add(x524, x490)
        return x525

m = M().eval()
x523 = torch.randn(torch.Size([1, 384, 768]))
x490 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x523, x490)
end = time.time()
print(end-start)
