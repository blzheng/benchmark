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
        self.dropout40 = Dropout(p=0.0, inplace=False)
        self.linear43 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout41 = Dropout(p=0.0, inplace=False)

    def forward(self, x497):
        x498=self.dropout40(x497)
        x499=self.linear43(x498)
        x500=self.dropout41(x499)
        return x500

m = M().eval()
x497 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x497)
end = time.time()
print(end-start)
