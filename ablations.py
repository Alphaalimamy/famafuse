import torch
import torch.nn as nn
import torch.nn.functional as F


# Spatial Attention Residual Module
class SARM(nn.Module):
    def __init__(self, in_channels:int, output_channels:int, bn_layer=True):
        super(SARM, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = output_channels

        # Initial convolution layers
        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels,
                          out_channels=self.in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
                nn.BatchNorm2d(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels,
                               out_channels=self.in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # Attention Layers
        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        # Spatial Pooling and Upsampling
        self.spatial_pool = nn.AdaptiveAvgPool2d((8, 8))  # Reduced spatial dimension
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)  # Restore original size

    def forward(self, v):
        '''
        :param v: (B, C, H, W)
        :return: Tensor of same size as input
        '''
        batch_size, c, H, W = v.size()

        # Apply spatial pooling
        v_pooled = self.spatial_pool(v)  # Downsample input

        g_v = self.g(v_pooled).view(batch_size, self.inter_channels, -1)  # (B, C', N)
        g_v = g_v.permute(0, 2, 1)  # (B, N, C')

        theta_v = self.theta(v_pooled).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)  # (B, N, C')

        phi_v = self.phi(v_pooled).view(batch_size, self.inter_channels, -1)

        # Attention Map
        L = torch.matmul(theta_v, phi_v)  # (B, N, N)
        N = L.size(-1)
        L_div_C = L / N

        # Attention Output
        y = torch.matmul(L_div_C, g_v)  # (B, N, C')
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v_pooled.size()[2:])

        W_y = self.W(y)

        # Upsample back to original size
        W_y = self.upsample(W_y)  # Restore original size

        # Residual connection and output
        v_star = torch.sigmoid(W_y) * v
        return v_star


class MGAIM(nn.Module):
    def __init__(self, channels:int):
        super(MGAIM, self).__init__()
        self.channels = channels

        # Learnable parameters for Gaussian kernel
        self.mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # Learnable weights for cross-feature interactions (3x3 matrix)
        self.beta = nn.Parameter(torch.ones(3, 3))
        self.softmax = nn.Softmax(dim=1) # Normalization layer (softmax across rows of beta)

        # Convolution layers for multiscale feature extraction
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)

        # Fuse features after cross-interaction
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C, H, W) after adaptive cross-feature weighting.
        """
        batch_size, _, h, w = x.size()

        # Multiscale Feature Extraction
        f1 = self.conv1x1(x)  # Feature at 1x1 scale
        f2 = self.conv3x3(x)  # Feature at 3x3 scale
        f3 = self.conv5x5(x)  # Feature at 5x5 scale

        # Normalize learnable weights beta
        beta_normalized = self.softmax(self.beta)  # (3, 3)

        # Apply cross-feature interactions
        f1_prime = beta_normalized[0, 0] * f1 + beta_normalized[0, 1] * f2 + beta_normalized[0, 2] * f3
        f2_prime = beta_normalized[1, 0] * f1 + beta_normalized[1, 1] * f2 + beta_normalized[1, 2] * f3
        f3_prime = beta_normalized[2, 0] * f1 + beta_normalized[2, 1] * f2 + beta_normalized[2, 2] * f3

        # Concatenate updated features
        cross_weighted_features = torch.cat([f1_prime, f2_prime, f3_prime], dim=1)  # (B, C*3, H, W)

        # Fuse features into final output
        fused_features = self.fuse(cross_weighted_features)  # (B, C, H, W)

        # Calculate query (global mean per channel)
        q = fused_features.mean(dim=[2, 3], keepdim=True)

        # Generate key and value tensors using convolution
        k = fused_features
        v = fused_features

        # Gaussian Kernel Attention Calculation
        diff = (k - q - self.mu).pow(2)
        sigma = diff.mean(dim=[2, 3], keepdim=True)
        epsilon = torch.finfo(sigma.dtype).eps
        att_score = diff / (2 * sigma + epsilon)

        att_score = F.softmax(att_score, dim=-1)
        att_weight = F.sigmoid(att_score + 0.5)

        output = att_weight * v
        return output



class IMFF(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(IMFF, self).__init__()

        # 1x1 and 3x3 convolutions for each modality
        self.conv1x1_A = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_A = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv1x1_B = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_B = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv1x1_common = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_common = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, FAi, FBi):
        # Apply attention maps
        KiA = self.sigmoid(self.conv3x3_A(self.conv1x1_A(FAi)))
        KiB = self.sigmoid(self.conv3x3_B(self.conv1x1_B(FBi)))
        KiC = self.sigmoid(self.conv3x3_common(self.conv1x1_common(FAi + FBi)))

        # Refine features
        F_hat_Ai = self.conv3x3_A(self.conv1x1_A(FAi)) + KiA * KiC
        F_hat_Bi = self.conv3x3_B(self.conv1x1_B(FBi)) + KiB * KiC

        # Concatenate and fuse features
        FF_i = self.fusion_conv(torch.cat([F_hat_Ai, F_hat_Bi], dim=1))

        return FF_i



class FusionModelIMFF(nn.Module):
    def __init__(self, channels, out_channels):
        super(FusionModelIMFF, self).__init__()

        self.imff = IMFF(in_channels=1, out_channels=out_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        #
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Args:
            x1: First input tensor (B, C, H, W)
            x2: Second input tensor (B, C, H, W)
        Returns:
            Combined output tensor (B, C, H, W)
        """
        imff = self.imff(x1, x2)
        fused_out = self.fusion_conv(imff)
        output = self.final_conv(fused_out)
        return output




class FusionModelSARM(nn.Module):
    def __init__(self, channels, out_channels):
        super(FusionModelSARM, self).__init__()


        self.sarm= SARM(in_channels=channels * 2, output_channels=out_channels)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        #
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Args:
            x1: First input tensor (B, C, H, W)
            x2: Second input tensor (B, C, H, W)
        Returns:
            Combined output tensor (B, C, H, W)
        """

        fused_out = torch.cat([x1, x2], dim=1)
        sarm = self.sarm(fused_out)
        fused_out = self.fusion_conv(sarm)
        output = self.final_conv(fused_out)
        return output




class FusionModelMGAIM(nn.Module):
    def __init__(self, channels, out_channels):
        super(FusionModelMGAIM, self).__init__()


        self.mgaim_1 = MGAIM(channels=2 * channels)
        self.mgaim_2 = MGAIM(channels=2 * channels)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(4 * channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Args:
            x1: First input tensor (B, C, H, W)
            x2: Second input tensor (B, C, H, W)
        Returns:
            Combined output tensor (B, C, H, W)
        """
        fused_out = torch.cat([x1, x2], dim=1)
        mgaim_1 = self.mgaim_1(fused_out)
        mgaim_2 = self.mgaim_2(fused_out)

        out = torch.cat([mgaim_1, mgaim_2], dim=1)
        fused_out = self.fusion_conv(out)
        output = self.final_conv(fused_out)
        return output

def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

