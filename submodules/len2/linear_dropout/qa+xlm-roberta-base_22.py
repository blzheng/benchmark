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
        self.linear69 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout35 = Dropout(p=0.1, inplace=False)

    def forward(self, x522):
        x523=self.linear69(x522)
        x524=self.dropout35(x523)
        return x524

m = M().eval()
x522 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x522)
end = time.time()
print(end-start)