import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class UNet(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):  # Changes num_classes to 1!
        super(UNet, self).__init__()
        self.backbone = backbone
        """
        Encoder:
        - Repeated Conv (3 x 3) -> Norm -> ReLU blocks
        - Downsample (MaxPool 2x2 or Stride-2 Conv)
        """

        # Function: Two convolutions (3x3) followed by ReLU Activation: One: (Edges, Corners) Two: (Complex Shapes)
        # In_channels: How many channels from image (RBG, Grayscale)
        # Out_channels: How many filters to produce (Feature Maps)
        def DoubleConv(in_channels, out_channels):
            return nn.Sequential(
                # First Convolution Layer: Extract patterns (Small shapes: Edges, Corners, etc..)
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3,  # 3 x 3 patch
                          padding=1),
                # Normalize output
                nn.BatchNorm2d(out_channels),
                # Add non-linearity for complex features f(x) = max(0, x)
                nn.ReLU(inplace=True),
                # Repeat: Second Convolution Layer
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        if self.backbone == 'scratch':
            # Downsample:  Image size halves from pooling; Channels double each level to capture context
            self.pool = nn.MaxPool2d(2, 2)  # Take Max value in patch, place in feature map
            self.down_conv1 = DoubleConv(3, 64)  # Takes in_channels -> 64 feature maps
            self.down_conv2 = DoubleConv(64, 128)  # Takes 64 -> 128 feature maps
            self.down_conv3 = DoubleConv(128, 256)  # Takes 128 -> 256 feature maps
            self.down_conv4 = DoubleConv(256, 512)  # Takes 256 -> 512 feature maps
            # Bottleneck: Deepest layer (another DoubleConv Block); Grasps global (image level) information
            self.bottleneck = DoubleConv(512, 1024)

            """
            Decoder:
            - Upsample by bilinear/nearest + Conv(3x3) or transposed conv
            - Channels halve each level
            - Add small refinement block after each upsampling step to sharpen boundaries
            """
            # Upsample layer: By transposed conv
            self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.decoder4 = DoubleConv(1024, 512)
            self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.decoder3 = DoubleConv(512, 256)
            self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.decoder2 = DoubleConv(256, 128)
            self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.decoder1 = DoubleConv(128, 64)


        elif self.backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.conv_skip = nn.Sequential(resnet.conv1,
                                           resnet.bn1,
                                           resnet.relu)
            self.pool1 = resnet.maxpool
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4

            """
            # Freeze & Unfreeze
 
            # Variant 2: ResUNet (Entire encoder Frozen)
            for param in resnet.parameters():
                param.requires_grad = False
            resnet.eval()
 
            # Variant 3: ResUNet (fine-tune all); Train encoder and decoder
 
            # Variant 4: ResUNet (partial fine-tune); Train last encoder stage and decoder, keep earlier enc stages frozen
            for param in resnet.parameters():
                param.requires_grad = False
            resnet.eval()
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            resnet.layer4.train()
 
            # Variant 5: ResUNet (fine-tune all, augmentation, mixed loss)
 
            """

            # Upsample layer: By transposed conv
            self.up_conv1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
            self.decoder4 = DoubleConv(2048, 1024)
            self.up_conv2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.decoder3 = DoubleConv(1024, 512)
            self.up_conv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.decoder2 = DoubleConv(512, 256)
            self.up_conv4 = nn.ConvTranspose2d(256, 64, 2, stride=2)
            self.decoder1 = DoubleConv(128, 64)

        # 1 x 1 conv head
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    # Upsample
    # Small Helper Function:
    # S: Skip-connection feature map (From Encoder)
    # D: Decoder feature map (From upsampled)
    def Upsample(self, D, S, up_layer, conv_layer):
        D = up_layer(D)  # Upsample feature map to spatial size
        x = torch.cat([S, D], dim=1)  # Concatenate [N, Ce + Cd, H, W]
        x = conv_layer(x)  # Merge encoder/decoder info
        return x  # Return feature map

    def forward(self, x):
        if self.backbone == 'scratch':
            # Encoder

            down_1 = self.down_conv1(x)
            down_2 = self.down_conv2(self.pool(down_1))
            down_3 = self.down_conv3(self.pool(down_2))
            down_4 = self.down_conv4(self.pool(down_3))

            # Bottleneck
            b = self.bottleneck(self.pool(down_4))

            # Decoder
            """
            # Final upsampling step
            """
            up_4 = self.Upsample(b, down_4, self.up_conv1, self.decoder4)
            up_3 = self.Upsample(up_4, down_3, self.up_conv2, self.decoder3)
            up_2 = self.Upsample(up_3, down_2, self.up_conv3, self.decoder2)
            up_1 = self.Upsample(up_2, down_1, self.up_conv4, self.decoder1)
            out = self.out(up_1)
            return out
        else:
            ######################
            # ResUNet
            ######################
            # Encoder
            e0 = self.conv_skip(x)
            e1 = self.encoder1(self.pool1(e0))
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)

            # Decoder
            d4 = self.Upsample(e4, e3, self.up_conv1, self.decoder4)
            d3 = self.Upsample(d4, e2, self.up_conv2, self.decoder3)
            d2 = self.Upsample(d3, e1, self.up_conv3, self.decoder2)
            d1 = self.Upsample(d2, e0, self.up_conv4, self.decoder1)

            out = self.out(d1)
            return F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)

