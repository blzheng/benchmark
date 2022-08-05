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
        self.dropout24 = Dropout(p=0.0, inplace=False)
        self.linear27 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout25 = Dropout(p=0.0, inplace=False)

    def forward(self, x313):
        x314=self.dropout24(x313)
        x315=self.linear27(x314)
        x316=self.dropout25(x315)
        return x316

m = M().eval()
x313 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x313)
end = time.time()
print(end-start)
