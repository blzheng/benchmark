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

    def forward(self, x422, x417):
        x423=torch.nn.functional.softmax(x422,dim=-1, _stacklevel=3, dtype=None)
        x424=self.dropout1(x423)
        x425=torch.matmul(x424, x417)
        return x425

m = M().eval()
x422 = torch.randn(torch.Size([1, 12, 384, 384]))
x417 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x422, x417)
end = time.time()
print(end-start)
