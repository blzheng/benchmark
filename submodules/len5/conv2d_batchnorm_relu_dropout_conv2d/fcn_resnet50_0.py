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
        self.conv2d53 = Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU()
        self.dropout0 = Dropout(p=0.1, inplace=False)
        self.conv2d54 = Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x174):
        x175=self.conv2d53(x174)
        x176=self.batchnorm2d53(x175)
        x177=self.relu49(x176)
        x178=self.dropout0(x177)
        x179=self.conv2d54(x178)
        return x179

m = M().eval()
x174 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
