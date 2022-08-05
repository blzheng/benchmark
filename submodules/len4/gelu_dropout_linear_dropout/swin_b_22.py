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
        self.gelu22 = GELU(approximate='none')
        self.dropout44 = Dropout(p=0.0, inplace=False)
        self.linear48 = Linear(in_features=4096, out_features=1024, bias=True)
        self.dropout45 = Dropout(p=0.0, inplace=False)

    def forward(self, x550):
        x551=self.gelu22(x550)
        x552=self.dropout44(x551)
        x553=self.linear48(x552)
        x554=self.dropout45(x553)
        return x554

m = M().eval()
x550 = torch.randn(torch.Size([1, 7, 7, 4096]))
start = time.time()
output = m(x550)
end = time.time()
print(end-start)
