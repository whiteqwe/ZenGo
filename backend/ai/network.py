import torch
import torch.nn as nn
import torch.nn.functional as F

class SEResBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation (SE) Channel Attention.
    Significantly enhances pattern recognition across distant board coordinates (KataGo architecture).
    """
    def __init__(self, channels):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-Excitation (SE) Attention
        reduction = max(4, channels // 4)
        self.se_fc1 = nn.Linear(channels, reduction)
        self.se_fc2 = nn.Linear(reduction, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Global average pooling + channel excitation
        b, c, _, _ = out.size()
        w = F.adaptive_avg_pool2d(out, 1).view(b, c)
        w = F.relu(self.se_fc1(w))
        w = torch.sigmoid(self.se_fc2(w)).view(b, c, 1, 1)
        
        out = out * w + residual
        return F.relu(out)

class AlphaGoZeroNet(nn.Module):
    """
    Upgraded AlphaZero + KataGo 29-Channel SOTA Multi-Task Network:
    - 16 History Channels (T, T-1, ..., T-7 for both players)
    - 8 Explicit Tactical Liberty Planes (1, 2, 3, >=4 liberties for both players: Atari / Semeai / Life & Death)
    - 4 Direct Capture & Self-Atari Tactical Foresight Planes
    - 1 Turn Color Plane
    - SE-Attention Deep ResNet Backbone
    - 4 Multi-task heads: Policy Head + Value Head + Ownership Territory Head + Score Lead Head
    """
    def __init__(self, board_size=9, in_channels=29, num_filters=128, num_res_blocks=8):
        super(AlphaGoZeroNet, self).__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        
        # Initial convolution
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # SE-Residual Backbone
        self.res_blocks = nn.ModuleList([SEResBlock(num_filters) for _ in range(num_res_blocks)])
        
        # 1. Policy Head (落子策略分布: 81 选点 + 1 虚手)
        self.policy_conv = nn.Conv2d(num_filters, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size + 1)
        
        # 2. Value Head (盘面胜率估值: [-1, 1])
        self.value_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # 3. Ownership Head (KataGo 领地归属头: 预测 81 个交叉点终局死活与地盘归属 [-1, 1])
        self.ownership_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=True)

        # 4. Score Lead Head (KataGo 精确目数差头: 预测终局领先目数 [-81, 81])
        self.score_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.score_bn = nn.BatchNorm2d(2)
        self.score_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.score_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
            
        # 1. Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)
        policy = F.log_softmax(policy, dim=1)
        
        # 2. Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        # 3. Ownership
        ownership = torch.tanh(self.ownership_conv(x)).squeeze(1) # (B, H, W)

        # 4. Score Lead
        s = F.relu(self.score_bn(self.score_conv(x)))
        s = s.view(s.size(0), -1)
        s = F.relu(self.score_fc1(s))
        score_lead = self.score_fc2(s)
        
        return policy, value, ownership, score_lead

    def predict(self, board_features, device="cpu"):
        """Utility for inference with FP16 Tensor Core acceleration"""
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(board_features, dtype=torch.float32, device=device).unsqueeze(0)
            if str(device).startswith("cuda"):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    log_policy, value, ownership, score_lead = self.forward(x)
            else:
                log_policy, value, ownership, score_lead = self.forward(x)
            policy = torch.exp(log_policy.float()).squeeze(0).cpu().numpy()
            value = float(value.item())
            ownership_grid = ownership.float().squeeze(0).cpu().numpy()
            lead = float(score_lead.item())
            return policy, value, ownership_grid, lead
