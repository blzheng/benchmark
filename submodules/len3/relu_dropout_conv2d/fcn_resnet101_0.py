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
        self.relu100 = ReLU()
        self.dropout0 = Dropout(p=0.1, inplace=False)
        self.conv2d105 = Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x346):
        x347=self.relu100(x346)
        x348=self.dropout0(x347)
        x349=self.conv2d105(x348)
        return x349

m = M().eval()
x346 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x346)
end = time.time()
print(end-start)
