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
        self.dropout16 = Dropout(p=0.1, inplace=False)

    def forward(self, x261, x251):
        x262=torch.nn.functional.softmax(x261,dim=-1, _stacklevel=3, dtype=None)
        x263=self.dropout16(x262)
        x264=torch.matmul(x263, x251)
        return x264

m = M().eval()
x261 = torch.randn(torch.Size([1, 12, 384, 384]))
x251 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x261, x251)
end = time.time()
print(end-start)
