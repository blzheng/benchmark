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
        self.dropout22 = Dropout(p=0.1, inplace=False)

    def forward(self, x345, x335):
        x346=torch.nn.functional.softmax(x345,dim=-1, _stacklevel=3, dtype=None)
        x347=self.dropout22(x346)
        x348=torch.matmul(x347, x335)
        return x348

m = M().eval()
x345 = torch.randn(torch.Size([1, 12, 384, 384]))
x335 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x345, x335)
end = time.time()
print(end-start)
