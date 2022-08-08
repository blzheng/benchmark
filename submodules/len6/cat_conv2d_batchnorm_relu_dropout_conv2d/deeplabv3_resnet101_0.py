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
        self.conv2d109 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d109 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu105 = ReLU()
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.conv2d110 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x347, x350, x353, x356, x363):
        x364=torch.cat([x347, x350, x353, x356, x363],dim=1)
        x365=self.conv2d109(x364)
        x366=self.batchnorm2d109(x365)
        x367=self.relu105(x366)
        x368=self.dropout0(x367)
        x369=self.conv2d110(x368)
        return x369

m = M().eval()
x347 = torch.randn(torch.Size([1, 256, 28, 28]))
x350 = torch.randn(torch.Size([1, 256, 28, 28]))
x353 = torch.randn(torch.Size([1, 256, 28, 28]))
x356 = torch.randn(torch.Size([1, 256, 28, 28]))
x363 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x347, x350, x353, x356, x363)
end = time.time()
print(end-start)
