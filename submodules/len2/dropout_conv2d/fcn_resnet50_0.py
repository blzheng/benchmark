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
        self.dropout0 = Dropout(p=0.1, inplace=False)
        self.conv2d54 = Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x177):
        x178=self.dropout0(x177)
        x179=self.conv2d54(x178)
        return x179

m = M().eval()
x177 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
