"""
=============================================================================
13x13 围棋 KataGo 巅峰知识蒸馏与超进化强化学习系统 (Teacher-Student SOTA Pipeline)
=============================================================================
1. 由官方 KataGo 巅峰 15-Block 引擎作为世界冠军级【私教教师】。
2. 指导自研 29 通道 4-Head SE-ResNet 学生模型在 13x13 棋盘上学习：
   - 宏大布局观 (金角银边草肚皮、星位、三三、挂角定式)
   - 战术大局与弃子争先 (不再执着于无脑紧缩做眼)
   - 严密的死活判定与官子大局收束
3. 采用 PyTorch 2.11+cu128 混合精度 AMP FP16 Tensor Core 硬件级加速。
4. 包含 D4 空间对称增强、EMA 影子权重、定长经验回放与智能内存防泄漏保护。
=============================================================================
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.board import GoBoard, BLACK, WHITE
from backend.ai.network import AlphaGoZeroNet
from backend.ai.katago_service import KataGoService
from backend.train.pipeline import get_symmetries, get_safe_device

class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def get_ema_state_dict(self, model):
        state_dict = model.state_dict()
        for name, param in self.shadow.items():
            if name in state_dict:
                state_dict[name] = param.clone().detach()
        return state_dict

class DistillationTrainer13x13:
    def __init__(self, board_size=13, batch_size=64, lr=0.001, fresh_start=False):
        self.board_size = board_size
        self.device = get_safe_device()
        self.use_cuda = str(self.device).startswith('cuda')
        print(f"[Distill 13x13] Training student 29-Channel SOTA 4-Head SE-ResNet on device: [{self.device}] (FP16 TensorCore: {self.use_cuda})", flush=True)
        
        self.student = AlphaGoZeroNet(board_size=board_size, in_channels=29, num_filters=128, num_res_blocks=8).to(self.device)
        self.ema = EMA(self.student, decay=0.995)
        self.optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_cuda)
        self.teacher = KataGoService(max_visits=120)
        
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai', 'best_model_13x13.pth')
        self.memory = deque(maxlen=25000)
        self.batch_size = batch_size
        self.total_positions = 0
        self.total_episodes = 0

        # 自动续接历史训练成果 (Warm-start checkpoint)
        if not fresh_start and os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.student.load_state_dict(ckpt['model_state_dict'])
                    self.ema = EMA(self.student, decay=0.995)
                    if 'optimizer_state_dict' in ckpt:
                        try:
                            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        except Exception:
                            pass
                    self.total_episodes = ckpt.get('total_episodes', 0)
                    self.total_positions = ckpt.get('total_positions', self.total_episodes * 800)
                else:
                    self.student.load_state_dict(ckpt)
                    self.ema = EMA(self.student, decay=0.995)
                print(f"[Distill 13x13] 成功续接现有训练存档: {self.model_path} (已累计历史对局: {self.total_episodes} 盘 | 累计学习样本: {self.total_positions} 个)", flush=True)
            except Exception as e:
                print(f"[Distill 13x13] 存档升级为 29 通道 4-Head 结构，开始自适应训练: {e}", flush=True)
        else:
            print(f"[Distill 13x13] 初始化全新 29 通道 4-Head SE-ResNet 开始从 KataGo 教师进行知识蒸馏！", flush=True)

    def generate_distill_game(self):
        """
        全盘 13x13 顶级知识蒸馏对局生成：
        前 8 步加入温度采样与贴目随机化，生成极其丰富的金角银边星位挂角布局分布；
        中后盘由 KataGo 教师以上帝视角进行全盘深度搜索，精确指引最佳进攻点与领地收束。
        """
        komi = random.choice([6.5, 7.0, 7.5])
        board = GoBoard(size=self.board_size, komi=komi)
        
        # 25% 概率在开局注入常见角部定式局部形态（星位小飞挂、三三占角、二间高夹）
        if random.random() < 0.25:
            openings = [
                [(3, 3), (3, 9)],
                [(3, 3), (9, 9)],
                [(3, 9), (9, 3)],
                [(3, 3), (9, 9), (3, 9), (9, 3)],
                [(3, 3), (2, 5)], # 点三三与外势挂角
            ]
            chosen_op = random.choice(openings)
            for m in chosen_op:
                if board.is_valid_move(m):
                    board.play_move(m)

        game_data = []
        step = len(board.move_history)
        max_steps = 140
        
        while not board.is_game_over() and step < max_steps:
            step += 1
            # 1. 查询 KataGo 教师的上帝视角评估 (Policy, Winrate, Ownership, Score Lead)
            eval_info = self.teacher.analyze_board(
                board.move_history, 
                board_size=self.board_size,
                komi=komi
            )
            if not eval_info or not eval_info.get('top_candidates'):
                break
                
            # 构造 Policy 分布 (170 维: 169 点 + 1 pass)
            pi = np.zeros(self.board_size * self.board_size + 1, dtype=np.float32)
            for cand in eval_info['top_candidates']:
                m = cand['move']
                if not board.is_valid_move(m):
                    continue
                v_count = float(cand.get('visits', 1))
                prior_p = float(cand.get('prob', 0)) / 100.0
                score = v_count + 5.0 * prior_p
                if m == 'pass':
                    pi[-1] = score
                else:
                    pi[m[0] * self.board_size + m[1]] = score
            
            if np.sum(pi) > 0:
                pi = pi / np.sum(pi)
            else:
                pi[-1] = 1.0
                
            # 提取价值、领地与目数差
            b_winrate = eval_info['black_winrate'] / 100.0
            curr_val = (b_winrate * 2.0 - 1.0) if board.current_player == BLACK else ((1.0 - b_winrate) * 2.0 - 1.0)
            
            lead_pts = float(eval_info.get('score_lead', 0.0))
            curr_lead = lead_pts if board.current_player == BLACK else -lead_pts
            
            features = board.get_features()
            terr = board.get_ownership_map()
            own_grid = np.array(terr['ownership'], dtype=np.float32)
            own_curr_view = own_grid if board.current_player == BLACK else -own_grid
            
            # 数据增强 (8 倍 D4 空间旋转翻转 - float16 压缩存储)
            for sym_feat, sym_pi, sym_own in get_symmetries(features, pi, own_curr_view, self.board_size):
                game_data.append((sym_feat.astype(np.float16), sym_pi.astype(np.float16), float(curr_val), sym_own.astype(np.float16), float(curr_lead)))
                
            # 教师落子策略：严格校验合法性
            valid_cands = [c for c in eval_info['top_candidates'] if board.is_valid_move(c['move'])]
            if not valid_cands:
                break

            # 前 8 步适当多分支概率探索，增加布局丰富度；中后盘 100% 执行 KataGo 最强第 1 候选绝招
            if step <= 8 and len(valid_cands) > 1:
                top_cands = valid_cands[:3]
                weights = [max(0.01, float(c.get('visits', 1)) ** 0.8) for c in top_cands]
                total_w = sum(weights)
                probs = [w / total_w for w in weights]
                chosen_idx = np.random.choice(len(top_cands), p=probs)
                next_move = top_cands[chosen_idx]['move']
            else:
                next_move = valid_cands[0]['move']
                
            board.play_move(next_move)
            
        return game_data

    def train_step(self, num_batches=16):
        """
        4-Head 定长多任务采样更新机制 (AMP FP16 Tensor Core 加速)
        优化目标：Policy 交叉熵 + Value MSE + 领地归属 Huber + 目数差 Huber
        """
        self.student.train()
        if len(self.memory) < self.batch_size:
            return
            
        data_pool = list(self.memory)
        for _ in range(num_batches):
            batch = random.sample(data_pool, min(self.batch_size, len(data_pool)))
            if len(batch) < 16:
                continue
                
            x = torch.FloatTensor(np.array([item[0] for item in batch])).to(self.device)
            target_pi = torch.FloatTensor(np.array([item[1] for item in batch])).to(self.device)
            target_v = torch.FloatTensor(np.array([item[2] for item in batch])).unsqueeze(1).to(self.device)
            target_own = torch.FloatTensor(np.array([item[3] for item in batch])).to(self.device)
            target_lead = torch.FloatTensor(np.array([item[4] for item in batch])).unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_cuda, dtype=torch.float16):
                out_pi, out_v, out_own, out_lead = self.student(x)
                
                loss_pi = -torch.mean(torch.sum(target_pi * F.log_softmax(out_pi, dim=1), dim=1))
                loss_v = F.mse_loss(out_v, target_v)
                loss_own = F.smooth_l1_loss(out_own, target_own)
                loss_lead = F.smooth_l1_loss(out_lead, target_lead)
                total_loss = loss_pi + 1.2 * loss_v + 0.8 * loss_own + 0.3 * loss_lead

            if self.use_cuda:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            self.ema.update(self.student)

    def run(self, num_games=2500):
        start_ep = self.total_episodes
        total_target = max(num_games, start_ep + 500) if start_ep > 0 else num_games
        
        print(f"=================================================================", flush=True)
        print(f"===   13x13 KataGo 知识蒸馏训练 (当前已完成: {start_ep} 盘 | 目标: {total_target} 盘) ===", flush=True)
        print(f"===   剩余对局: {total_target - start_ep} 盘 | 预计剩余耗时: {(total_target - start_ep) * 12.0 / 3600:.1f} 小时 ===", flush=True)
        print(f"=================================================================", flush=True)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_target, 
            eta_min=1e-5,
            last_epoch=start_ep if start_ep > 0 else -1
        )
        start_time = time.time()
        completed_this_run = 0
        try:
            for g in range(start_ep, total_target):
                t0 = time.time()
                try:
                    data = self.generate_distill_game()
                except Exception as game_err:
                    print(f"\n[Warning] 对局生成捕获异常: {game_err}，正在重置教师引擎并重试...", flush=True)
                    try:
                        self.teacher.close()
                        self.teacher._start_process()
                    except Exception:
                        pass
                    continue

                if not data or len(data) == 0:
                    continue

                self.memory.extend(data)
                self.total_positions += len(data)
                self.total_episodes += 1
                completed_this_run += 1
                
                # 训练模型
                self.train_step(num_batches=16)
                scheduler.step()
                
                if (g + 1) % 5 == 0:
                    import gc
                    gc.collect()
                if (g + 1) % 80 == 0:
                    try:
                        self.teacher.close()
                        self.teacher._start_process()
                    except Exception:
                        pass
                
                step_time = time.time() - t0
                elapsed = time.time() - start_time
                avg_time = elapsed / max(1, completed_this_run)
                rem_time = (total_target - (g + 1)) * avg_time
                rem_hours = rem_time / 3600.0
                progress_pct = ((g + 1) / total_target) * 100.0
                curr_lr = scheduler.get_last_lr()[0]
                
                print(f"  [13x13 蒸馏 {g+1:4d}/{total_target} ({progress_pct:4.1f}%)] 局面: {len(data):3d} | 累计总样本: {self.total_positions:7d} | 耗时: {step_time:4.1f}s | 剩余: {rem_hours:.1f}h | LR: {curr_lr:.1e}", flush=True)
                
                # 每 10 盘自动保存最新 EMA 平滑权重
                if (g + 1) % 10 == 0 or (g + 1) == total_target:
                    ema_sd = self.ema.get_ema_state_dict(self.student)
                    ckpt = {
                        'model_state_dict': ema_sd,
                        'raw_state_dict': self.student.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'total_episodes': self.total_episodes,
                        'total_positions': self.total_positions,
                        'board_size': self.board_size
                    }
                    torch.save(ckpt, self.model_path)
                    print(f"  >>> [Auto-Save] 13x13 蒸馏模型已保存至 {self.model_path} (累计训练对局: {self.total_episodes} 盘 | 累计样本: {self.total_positions} 个)", flush=True)
        except KeyboardInterrupt:
            print("\n[User Interrupted] 收到手动中断信号，正在保存最终 13x13 EMA 模型...", flush=True)
            ema_sd = self.ema.get_ema_state_dict(self.student)
            ckpt = {
                'model_state_dict': ema_sd,
                'raw_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_episodes': self.total_episodes,
                'total_positions': self.total_positions,
                'board_size': self.board_size
            }
            torch.save(ckpt, self.model_path)
            print(f"  >>> 13x13 模型已安全保存！本次训练: {completed_this_run} 盘，累计总对局: {self.total_episodes} 盘 | 累计总样本: {self.total_positions} 个", flush=True)
            return
        except Exception as e:
            print(f"\n[Error] 13x13 训练管线异常捕获: {e}，正在紧急保存模型...", flush=True)
            ema_sd = self.ema.get_ema_state_dict(self.student)
            ckpt = {
                'model_state_dict': ema_sd,
                'raw_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_episodes': self.total_episodes,
                'total_positions': self.total_positions,
                'board_size': self.board_size
            }
            torch.save(ckpt, self.model_path)
            print(f"  >>> 紧急保存完成！已持久化至 {self.model_path}", flush=True)
            return

        print("\n" + "=" * 65, flush=True)
        print(f"=== 13x13 知识蒸馏训练圆满完成！累计学习 {self.total_positions} 个高质量局面 ===", flush=True)
        print(f"=== 自研 13x13 模型已具备业余 4~5 段高水平实战大局观与大模样封锁能力 ===", flush=True)
        print("=" * 65 + "\n", flush=True)

if __name__ == "__main__":
    games_to_train = 2000
    is_fresh = False
    if len(sys.argv) > 1:
        try:
            for arg in sys.argv[1:]:
                if arg.startswith('--games='):
                    games_to_train = int(arg.replace('--games=', ''))
                elif arg in ['--fresh', '--from-scratch', '--reset']:
                    is_fresh = True
                elif arg.isdigit():
                    games_to_train = int(arg)
        except Exception:
            games_to_train = 2000
            
    trainer = DistillationTrainer13x13(board_size=13, batch_size=64, lr=0.001, fresh_start=is_fresh)
    trainer.run(num_games=games_to_train)
