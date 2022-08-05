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
        self.linear54 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout27 = Dropout(p=0.1, inplace=False)

    def forward(self, x403, x401):
        x404=self.linear54(x403)
        x405=self.dropout27(x404)
        x406=operator.add(x405, x401)
        return x406

m = M().eval()
x403 = torch.randn(torch.Size([1, 384, 1024]))
x401 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x403, x401)
end = time.time()
print(end-start)
