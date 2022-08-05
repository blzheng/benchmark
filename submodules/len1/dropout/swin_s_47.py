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
        self.dropout47 = Dropout(p=0.0, inplace=False)

    def forward(self, x576):
        x577=self.dropout47(x576)
        return x577

m = M().eval()
x576 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x576)
end = time.time()
print(end-start)
