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
        self.dropout25 = Dropout(p=0.1, inplace=False)

    def forward(self, x389, x378):
        x390=self.dropout25(x389)
        x391=torch.matmul(x390, x378)
        return x391

m = M().eval()
x389 = torch.randn(torch.Size([1, 4, 384, 384]))
x378 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x389, x378)
end = time.time()
print(end-start)
