import torch
import torch.nn as nn
from timm import create_model
import math

class ConvAndPooling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvAndPooling, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class ModifiedSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ModifiedSelfAttention, self).__init__()
        self.conv_key = ConvAndPooling(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_value = ConvAndPooling(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_query = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        batch_size = query.size(0)

        query_flat = query.view(batch_size, query.size(1), -1).transpose(1, 2)
        key = key.view(batch_size, key.size(1))
        value = value.view(batch_size, value.size(1))

        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        attention_scores = torch.matmul(query_flat, key.transpose(-1, -2))
        attention_probs = torch.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_probs, value)
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, attended_values.size(1),
                                              int(math.sqrt(attended_values.size(2))),
                                              int(math.sqrt(attended_values.size(2))))

        return attended_values

class FeatureFusion(nn.Module):
    def __init__(self, num_features=4):
        super(FeatureFusion, self).__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_features) / num_features)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, features_list):
        weights = self.softmax(self.attention_weights)
        weighted_sum = 0
        for i, feature in enumerate(features_list):
            weighted_sum += feature * weights[i]
        return weighted_sum

class FPNWithAttention(nn.Module):
    def __init__(self, in_channels):
        super(FPNWithAttention, self).__init__()
        self.conv1 = ConvAndPooling(in_channels, 256, kernel_size=1)
        self.conv3 = ConvAndPooling(in_channels, 256, kernel_size=3, padding=1)
        self.conv5 = ConvAndPooling(in_channels, 256, kernel_size=5, padding=2)
        self.conv7 = ConvAndPooling(in_channels, 256, kernel_size=7, padding=3)
        self.feature_fusion = FeatureFusion(num_features=4)

    def forward(self, x):
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        feat7 = self.conv7(x)

        features_list = [feat1, feat3, feat5, feat7]
        fused_feature = self.feature_fusion(features_list)
        combined_features = torch.cat(features_list, dim=1)

        return combined_features, fused_feature

class SwinWithAttentionFusion(nn.Module):
    def __init__(self, num_classes, device='cuda'):
        super(SwinWithAttentionFusion, self).__init__()
        self.swin_transformer = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)
        self.swin_backbone = self.swin_transformer.forward_features
        self.fpn = FPNWithAttention(in_channels=1024)
        self.fc_concat = nn.Linear(1024, num_classes)
        self.fc_attention = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.weight_concat = nn.Parameter(torch.tensor(0.5))
        self.weight_attention = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        swin_out = self.swin_backbone(x)
        swin_out = swin_out.permute(0, 3, 1, 2)

        if len(swin_out.shape) == 3:
            B, L, C = swin_out.shape
            H = W = int(math.sqrt(L))
            swin_out = swin_out.reshape(B, L, C).permute(0, 2, 1).reshape(B, C, H, W)

        fpn_concat, fpn_fusion = self.fpn(swin_out)

        concat_feature = fpn_concat.view(fpn_concat.size(0), -1)
        fusion_feature = fpn_fusion.view(fpn_fusion.size(0), -1)

        concat_feature = self.dropout(concat_feature)
        fusion_feature = self.dropout(fusion_feature)

        out_concat = self.fc_concat(concat_feature)
        out_attention = self.fc_attention(fusion_feature)

        weight_sum = self.weight_concat + self.weight_attention
        normalized_weight_concat = self.weight_concat / weight_sum
        normalized_weight_attention = self.weight_attention / weight_sum

        out = normalized_weight_concat * out_concat + normalized_weight_attention * out_attention

        return out


def load_model(model_path, num_classes=202, device='cpu'):
    """加载训练好的模型"""
    print(f"🔄 正在加载模型: {model_path}")
    print(f"🖥️  使用设备: {device}")
    
    model = SwinWithAttentionFusion(num_classes=num_classes, device=device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 模型加载成功！")
                print(f"   训练轮次: {checkpoint.get('epoch', 'Unknown')}")
                print(f"   测试准确率: {checkpoint.get('test_top1_acc', 'Unknown'):.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 模型加载成功！")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 模型加载成功！")
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise
    
    model.eval()
    return model
