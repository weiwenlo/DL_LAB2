import torch.nn

def conv3x3(in_channels, out_outchannels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_channels, out_outchannels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_outchannels, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_channels, out_outchannels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(torch.nn.Module):
    expansion: int = 1
    def __init__(self,in_channels, out_channels,stride=1,downsample=None, 
                 groups=1,base_width=64,dilation=1,norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels,stride=1)
        self.bn2 = norm_layer(out_channels)

        # downsample 是為了讓shortcut 的channels數 和一班path 一致 
        self.downsample = downsample
        self.stride = stride
    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
class BottleNeck(torch.nn.Module):
    def __init__(self,in_channels, out_channels,stride=1,downsample=None, 
                norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels,stride=1)
        self.bn2 = norm_layer(out_channels)

        # downsample 是為了讓shortcut 的channels數 和一班path 一致 
        self.downsample = torch.nn.Sequential(
                conv1x1(512, 256, stride),
                norm_layer(256),
            )
        self.stride = stride
    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
class DoubleConv(torch.nn.Module):
    def __init__ (self, in_channels , out_channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3,stride=1, padding=1,bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3,stride=1, padding=1,bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),

        )
    def forward(self, x):
        return self.conv(x)
class ResNetU(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, 
                 width_per_group=64,  norm_layer=None):
        super().__init__()
        #  first conv 
        self.conv_outchannels =64 #第一次 conv 要求64 channels

        ## when s=2 , kernel 7 => padding 3   
        ## when s=2 , kernel 5 => padding 2     
        ## when s=2 , kernel 3 => padding 1       
        # => 依據o = (i-k+2*p)/s +1 ,outputsize會減半 , 
        # Q: k是奇數, 除s都會有小樹, 為何k不用 6,4,2? 
        # A: kernel 為奇數時 有中心點
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels= self.conv_outchannels, 
                                      kernel_size=7,stride=2, padding=3,bias=False)
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.bn1 = norm_layer(self.conv_outchannels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #############
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bottleneck0 =  torch.nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3,stride=2, padding=1)
        self.bottleneck = DoubleConv(512,1024)

        self.ups = torch.nn.ModuleList()
        for feature in [512,256,128,64]:
            self.ups.append(torch.nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.ups2 = torch.nn.ModuleList()
        for feature in [32,16]:
            self.ups2.append(torch.nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups2.append(DoubleConv(feature, feature))
        self.final_conv = torch.nn.Conv2d(in_channels=16, out_channels=num_classes
                                    ,kernel_size=1)
        #########
        self.base_width = width_per_group
    def forward(self,x):
        #  first conv 
        #print(f"1:{x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(f"2:{x.shape}")
        #############
        skip_connections = []
        x = self.layer1(x)
        #print(f"3:{x.shape}")
        skip_connections.append(x)
        x = self.layer2(x)
        #print(f"4:{x.shape}")
        skip_connections.append(x)
        x = self.layer3(x)
        #print(f"5:{x.shape}")
        skip_connections.append(x)
        x = self.layer4(x)
        #print(f"6:{x.shape}")
        skip_connections.append(x) 
        skip_connections.reverse()
        x = self.bottleneck0(x) #7:torch.Size([4, 512, 8, 8])
        x = self.bottleneck(x) #7:torch.Size([4, 512, 4, 4])
        #print(f"7:{x.shape}")

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            #print(f"s:{skip_connection.shape}")
            #print(f"loop {i}:{x.shape}")
            concat =torch.cat((skip_connection, x), dim=1) # N x C x H x W 把channel cat在一起
            x= self.ups[i+1](concat)
        for i in range(0, len(self.ups2), 2):
            x = self.ups2[i](x)
            x= self.ups2[i+1](x)
        #print(f"final:{x.shape}")
        return self.final_conv(x)
    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=False):
        """
        block: resnet34的basic block
        planes: block 內部的傳送時的channel數量
        blocks 一個layer有多少block
        """
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.conv_outchannels != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.conv_outchannels, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers= []
        #第一個block
        layers.append(block(self.conv_outchannels, planes, stride,downsample, 
                  norm_layer=norm_layer))
        self.conv_outchannels = planes * block.expansion
        for _ in range(1, blocks):#這邊處理第2個以後的blocks
            layers.append(block(self.conv_outchannels, planes, 
                norm_layer=norm_layer))
        return torch.nn.Sequential(*layers)