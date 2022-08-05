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
        self.linear10 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu4 = GELU(approximate='none')
        self.dropout8 = Dropout(p=0.0, inplace=False)

    def forward(self, x127):
        x128=self.linear10(x127)
        x129=self.gelu4(x128)
        x130=self.dropout8(x129)
        return x130

m = M().eval()
x127 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x127)
end = time.time()
print(end-start)
