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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x311, x306):
        x312=torch.nn.functional.softmax(x311,dim=-1, _stacklevel=3, dtype=None)
        x313=self.dropout1(x312)
        x314=torch.matmul(x313, x306)
        return x314

m = M().eval()
x311 = torch.randn(torch.Size([1, 12, 384, 384]))
x306 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x311, x306)
end = time.time()
print(end-start)
