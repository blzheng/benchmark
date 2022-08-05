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
        self.linear28 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu13 = GELU(approximate='none')
        self.dropout26 = Dropout(p=0.0, inplace=False)
        self.linear29 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout27 = Dropout(p=0.0, inplace=False)

    def forward(self, x334):
        x335=self.linear28(x334)
        x336=self.gelu13(x335)
        x337=self.dropout26(x336)
        x338=self.linear29(x337)
        x339=self.dropout27(x338)
        return x339

m = M().eval()
x334 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x334)
end = time.time()
print(end-start)
