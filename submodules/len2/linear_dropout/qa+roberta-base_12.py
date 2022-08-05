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
        self.linear39 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout20 = Dropout(p=0.1, inplace=False)

    def forward(self, x312):
        x313=self.linear39(x312)
        x314=self.dropout20(x313)
        return x314

m = M().eval()
x312 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x312)
end = time.time()
print(end-start)
