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
        self.relu19 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)

    def forward(self, x92):
        x93=self.relu19(x92)
        x94=self.conv2d29(x93)
        return x94

m = M().eval()
x92 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
