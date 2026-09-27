import os
import sys
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.board import GoBoard, BLACK, WHITE
from backend.ai.network import AlphaGoZeroNet
from backend.ai.mcts_alpha import MCTSAlpha, action_to_index, index_to_action
from backend.train.pipeline import get_symmetries, get_safe_device

class EMA:
    """Exponential Moving Average (EMA) for pure self-play weights."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param)

    def get_ema_state_dict(self, model):
        sd = model.state_dict()
        for name, param in self.shadow.items():
            if name in sd:
                sd[name] = param.clone()
        return sd

def worker_play_single_game(state_dict_bytes, board_size=13, worker_threads=3, device_str='cuda'):
    """
    独立工作进程：执行一盘 13x13 自对弈
    """
    try:
        import io
        if device_str == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            torch.set_num_threads(worker_threads)
        
        network = AlphaGoZeroNet(board_size=board_size, in_channels=29, num_filters=128, num_res_blocks=8).to(device)
        buffer = io.BytesIO(state_dict_bytes)
        sd = torch.load(buffer, map_location=device)
        network.load_state_dict(sd)
        network.eval()

        komi = random.choice([5.5, 6.5, 7.0, 7.5])
        board = GoBoard(size=board_size, komi=komi)
        
        # 社区 SOTA 1：20% 几率进行 2~4 步随机开局微扰 (Anti-Mode Collapse)
        if random.random() < 0.20:
            for _ in range(random.randint(2, 4)):
                legal = board.get_legal_moves()
                non_pass = [m for m in legal if m != 'pass']
                if non_pass:
                    board.play_move(random.choice(non_pass))

        # MCTS 树搜索 (纯自建，深度死活与全局推演)
        mcts_fast = MCTSAlpha(network, num_simulations=30, device=device)
        mcts_deep = MCTSAlpha(network, num_simulations=100, device=device)
        
        game_history = []
        step = 0
        max_steps = 140
        allow_resign = (random.random() < 0.85) # 85% 允许绝望大败时提前认输加速
        
        while not board.is_game_over() and step < max_steps:
            step += 1
            temp = 1.0 if step <= 10 else 0.2
            
            # Playout Cap Randomization 社区加速
            mcts = mcts_deep if (random.random() < 0.25 or step <= 8) else mcts_fast
            pi, root_val, _ = mcts.get_action_prob(board, temp=temp, add_noise=(step <= 10), return_search_info=True)
            
            # 动态绝望认输加速 (节约 50% 步数)
            if allow_resign and step >= 36:
                if root_val < -0.92:
                    board.resigned_player = board.current_player
                    break
            
            features = board.get_features()
            game_history.append((features, board.current_player, pi))
            
            if temp == 0.0:
                action_idx = int(np.argmax(pi))
            else:
                action_idx = int(np.random.choice(len(pi), p=pi))
                
            action = index_to_action(action_idx, board_size)
            board.play_move(action)
            
        # 终局裁判结算 (Tromp-Taylor 空间领地与目数差)
        if board.resigned_player is not None:
            winner = -board.resigned_player
            lead_pts = 30.0 if winner == BLACK else -30.0
            final_own = np.full((board_size, board_size), 1.0 if winner == BLACK else -1.0, dtype=np.float32)
        else:
            score_data = board.get_ownership_map()
            final_own = np.array(score_data["ownership"], dtype=np.float32)
            lead_pts = float(score_data.get("lead", 0.0))
            if lead_pts > 0:
                winner = BLACK
            elif lead_pts < 0:
                winner = WHITE
            else:
                winner = 0
                
        # 8 倍 D4 空间对称数据增强
        augmented_data = []
        for feat, player, p_dist in game_history:
            if winner == 0:
                v_target = 0.0
            else:
                v_target = 1.0 if player == winner else -1.0
                
            lead_target = lead_pts if player == BLACK else -lead_pts
            own_target = final_own if player == BLACK else -final_own
            
            for sym_feat, sym_pi, sym_own in get_symmetries(feat, p_dist, own_target, board_size):
                augmented_data.append((sym_feat.astype(np.float16), sym_pi.astype(np.float16), float(v_target), sym_own.astype(np.float16), float(lead_target)))
                
        return augmented_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

class PureSelfPlayTrainer13x13:
    """
    13x13 纯粹 AlphaZero 多进程并发自我进化系统
    【24 核心并行满载 · 零外部依赖 · 纯自建左右互搏】
    """
    def __init__(self, board_size=13, batch_size=128, lr=0.001, num_workers=4, worker_device=None):
        self.board_size = board_size
        self.device = get_safe_device()
        self.num_workers = num_workers
        self.use_cuda = str(self.device).startswith('cuda')
        
        # 智能设备分配：若并发数 <= 4 且有 GPU 则 Workers 使用 GPU；若并发数 > 4 则 Workers 自动分配至 CPU 避免显存溢出
        if worker_device:
            self.worker_device = worker_device
        else:
            self.worker_device = 'cuda' if (self.use_cuda and self.num_workers <= 4) else 'cpu'
            
        print(f"[Pure Self-Play 13x13] 启动多进程并发进化引擎 (并行工作进程: {self.num_workers} | 对弈设备: {self.worker_device.upper()} | 学习设备: {self.device} | FP16 TensorCore: {self.use_cuda})", flush=True)
        
        self.network = AlphaGoZeroNet(board_size=board_size, in_channels=29, num_filters=128, num_res_blocks=8).to(self.device)
        self.ema = EMA(self.network, decay=0.995)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_cuda)
        
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai', 'best_model_13x13.pth')
        self.memory = deque(maxlen=60000)
        self.batch_size = batch_size
        self.total_positions = 0
        self.total_episodes = 0

        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.network.load_state_dict(ckpt['model_state_dict'])
                    self.ema = EMA(self.network, decay=0.995)
                    if 'optimizer_state_dict' in ckpt:
                        try:
                            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        except Exception:
                            pass
                    self.total_episodes = ckpt.get('total_episodes', 0)
                    self.total_positions = ckpt.get('total_positions', self.total_episodes * 800)
                else:
                    self.network.load_state_dict(ckpt)
                    self.ema = EMA(self.network, decay=0.995)
                print(f"[Pure Self-Play 13x13] 成功续接历史存档: {self.model_path} (已累计自对弈: {self.total_episodes} 盘 | 累计样本: {self.total_positions} 个)", flush=True)
            except Exception as e:
                print(f"[Pure Self-Play 13x13] 初始化全新权重开始从零自我进化。", flush=True)
        else:
            print(f"[Pure Self-Play 13x13] 初始化全新权重开始从零自我进化。", flush=True)

    def train_step(self, num_batches=32):
        self.network.train()
        if len(self.memory) < self.batch_size:
            return
            
        data_pool = list(self.memory)
        for _ in range(num_batches):
            batch = random.sample(data_pool, min(self.batch_size, len(data_pool)))
            if len(batch) < 16:
                continue
                
            x = torch.as_tensor(np.array([item[0] for item in batch]), dtype=torch.float32, device=self.device)
            target_pi = torch.as_tensor(np.array([item[1] for item in batch]), dtype=torch.float32, device=self.device)
            target_v = torch.as_tensor(np.array([item[2] for item in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
            target_own = torch.as_tensor(np.array([item[3] for item in batch]), dtype=torch.float32, device=self.device)
            target_lead = torch.as_tensor(np.array([item[4] for item in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            if self.use_cuda:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out_pi, out_v, out_own, out_lead = self.network(x)
                    loss_pi = -torch.sum(target_pi * out_pi) / target_pi.size(0)
                    loss_v = F.mse_loss(out_v, target_v)
                    loss_own = F.smooth_l1_loss(out_own, target_own)
                    loss_lead = F.smooth_l1_loss(out_lead, target_lead)
                    total_loss = loss_pi + 1.2 * loss_v + 0.8 * loss_own + 0.3 * loss_lead
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out_pi, out_v, out_own, out_lead = self.network(x)
                loss_pi = -torch.sum(target_pi * out_pi) / target_pi.size(0)
                loss_v = F.mse_loss(out_v, target_v)
                loss_own = F.smooth_l1_loss(out_own, target_own)
                loss_lead = F.smooth_l1_loss(out_lead, target_lead)
                total_loss = loss_pi + 1.2 * loss_v + 0.8 * loss_own + 0.3 * loss_lead
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            self.ema.update(self.network)

    def run(self, num_games=2000):
        start_ep = self.total_episodes
        total_target = max(num_games, start_ep + 500) if start_ep > 0 else num_games
        
        print(f"=================================================================", flush=True)
        print(f"===   13x13 纯粹 AlphaZero 多进程并发进化强化学习系统         ===", flush=True)
        print(f"===   当前进度: {start_ep} 盘 | 目标总盘数: {total_target} 盘 | 并行度: {self.num_workers} 进程 ===", flush=True)
        print(f"=================================================================", flush=True)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_target, 
            eta_min=1e-5,
            last_epoch=start_ep if start_ep > 0 else -1
        )
        
        import io
        start_time = time.time()
        completed_count = start_ep
        batch_workers = self.num_workers
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            try:
                while completed_count < total_target:
                    t0 = time.time()
                    
                    # 序列化当前 EMA 权重分发给各个并行 Worker
                    ema_sd = self.ema.get_ema_state_dict(self.network)
                    bio = io.BytesIO()
                    torch.save(ema_sd, bio)
                    state_bytes = bio.getvalue()
                    
                    # 并行投递一批自对弈对局任务
                    current_batch_size = min(batch_workers, total_target - completed_count)
                    futures = [
                        executor.submit(worker_play_single_game, state_bytes, self.board_size, 2, str(self.worker_device))
                        for _ in range(current_batch_size)
                    ]
                    
                    batch_samples = 0
                    for fut in as_completed(futures):
                        game_data = fut.result()
                        if game_data and len(game_data) > 0:
                            self.memory.extend(game_data)
                            self.total_positions += len(game_data)
                            batch_samples += len(game_data)
                            completed_count += 1
                            self.total_episodes = completed_count
                    
                    # 集中训练梯度更新
                    self.train_step(num_batches=24)
                    for _ in range(current_batch_size):
                        scheduler.step()
                    import gc
                    gc.collect()
                        
                    batch_time = time.time() - t0
                    sec_per_game = batch_time / max(1, current_batch_size)
                    games_per_hour = 3600.0 / max(0.1, sec_per_game)
                    
                    elapsed = time.time() - start_time
                    completed_this_run = completed_count - start_ep
                    avg_time_per_game = elapsed / max(1, completed_this_run)
                    rem_seconds = (total_target - completed_count) * avg_time_per_game
                    rem_hours = rem_seconds / 3600.0
                    progress_pct = (completed_count / total_target) * 100.0
                    curr_lr = scheduler.get_last_lr()[0]
                    
                    print(f"  [并发自对弈 {completed_count:4d}/{total_target} ({progress_pct:4.1f}%)] +{current_batch_size}盘 | 局面: {batch_samples:4d} | 累计总样本: {self.total_positions:7d} | 速度: {sec_per_game:4.1f}s/盘 ({games_per_hour:4.0f}盘/h) | 剩余: {rem_hours:4.1f}h | LR: {curr_lr:.1e}", flush=True)
                    
                    # 每 10 盘自动落盘
                    if completed_count % 10 < current_batch_size or completed_count >= total_target:
                        ckpt = {
                            'model_state_dict': self.ema.get_ema_state_dict(self.network),
                            'raw_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'total_episodes': completed_count,
                            'total_positions': self.total_positions,
                            'board_size': self.board_size
                        }
                        torch.save(ckpt, self.model_path)
                        print(f"  >>> [Auto-Save] 并发进化模型已保存至 {self.model_path} (累计自对弈: {completed_count} 盘 | 累计样本: {self.total_positions} 个)", flush=True)
            except KeyboardInterrupt:
                print("\n[User Interrupted] 手动中断，正在保存当前自对弈进化成果...", flush=True)
                ckpt = {
                    'model_state_dict': self.ema.get_ema_state_dict(self.network),
                    'raw_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'total_episodes': completed_count,
                    'total_positions': self.total_positions,
                    'board_size': self.board_size
                }
                torch.save(ckpt, self.model_path)
                print(f"  >>> 纯自对弈模型已安全保存！累计对局: {completed_count} 盘", flush=True)
                return

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    games_to_train = 8000 # 默认整夜运行 8000 盘 (约 7.5~8.5 小时)
    workers = 4 # 默认 4 并发进程 (每进程 3 线程)，内存极其安全稳定 (总占用仅 ~50%)
    w_device = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--games='):
                games_to_train = int(arg.replace('--games=', ''))
            elif arg.startswith('--workers='):
                workers = int(arg.replace('--workers=', ''))
            elif arg.startswith('--worker-device='):
                w_device = arg.replace('--worker-device=', '').strip().lower()
            elif arg.isdigit():
                games_to_train = int(arg)
                
    trainer = PureSelfPlayTrainer13x13(board_size=13, batch_size=128, lr=0.001, num_workers=workers, worker_device=w_device)
    trainer.run(num_games=games_to_train)
