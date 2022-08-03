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
        self.dropout16 = Dropout(p=0.0, inplace=False)
        self.linear19 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x221):
        x222=self.dropout16(x221)
        x223=self.linear19(x222)
        return x223

m = M().eval()
x221 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x221)
end = time.time()
print(end-start)
