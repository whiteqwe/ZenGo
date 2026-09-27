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

# To allow relative imports if run directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.board import GoBoard, BLACK, WHITE
from backend.ai.network import AlphaGoZeroNet
from backend.ai.mcts_alpha import MCTSAlpha, action_to_index

def get_symmetries(features, pi, ownership_map, size=9):
    """
    Applies D4 dihedral group transformations (4 rotations x 2 reflections = 8 symmetries).
    features: (17, size, size)
    pi: (size*size + 1,) array of probabilities
    ownership_map: (size, size) array in [-1, 1]
    returns list of 8 (sym_features, sym_pi, sym_ownership) tuples
    """
    symmetries = []
    pi_board = np.array(pi[:-1]).reshape(size, size)
    pass_prob = pi[-1]

    for k in [0, 1, 2, 3]: # 0, 90, 180, 270 degree rotations
        r_features = np.rot90(features, k, (1, 2)).copy()
        r_pi_board = np.rot90(pi_board, k).copy()
        r_own = np.rot90(ownership_map, k).copy()
        sym_pi = np.array(list(r_pi_board.ravel()) + [pass_prob], dtype=np.float32)
        symmetries.append((r_features, sym_pi, r_own))

        # Horizontal flip
        f_features = np.flip(r_features, axis=2).copy()
        f_pi_board = np.fliplr(r_pi_board).copy()
        f_own = np.fliplr(r_own).copy()
        sym_f_pi = np.array(list(f_pi_board.ravel()) + [pass_prob], dtype=np.float32)
        symmetries.append((f_features, sym_f_pi, f_own))

    return symmetries

def get_safe_device():
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1, 1, device='cuda')
            _ = x + 1
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")

class AlphaZeroTrainer:
    def __init__(self, board_size=9, iters=100, eps=6, epochs=5, batch_size=128, lr=0.0005):
        self.board_size = board_size
        self.device = get_safe_device()
        self.network = AlphaGoZeroNet(board_size=board_size, in_channels=29, num_filters=128, num_res_blocks=8).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.iters = iters
        self.eps = eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=100000)
        
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai', 'best_model.pth')
        self.total_episodes = 0
        self.start_iteration = 0
        
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.network.load_state_dict(ckpt['model_state_dict'])
                    if 'optimizer_state_dict' in ckpt:
                        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    self.total_episodes = ckpt.get('total_episodes', 0)
                    self.start_iteration = ckpt.get('iteration', 0)
                else:
                    self.network.load_state_dict(ckpt)
                
                print("=" * 65)
                print(f"[AlphaZero] Loaded checkpoint: {self.model_path} (Games: {self.total_episodes})")
                print("=" * 65)
            except Exception as e:
                print(f"[AlphaZero] Checkpoint mismatch/upgrade: {e}. Starting fresh 17-channel SE-ResNet.")
        else:
            print("[AlphaZero] Starting fresh 17-channel SE-ResNet training.")
        
    def execute_episode(self):
        board = GoBoard(size=self.board_size)
        mcts = MCTSAlpha(self.network, num_simulations=60, device=self.device)
        train_examples = []
        
        step_count = 0
        while True:
            step_count += 1
            
            # 开局前 10 步高温探索 (temp=1.0)，中后盘低温精准选择 (temp=0.2)
            temp = 1.0 if step_count < 10 else 0.2
            pi = mcts.get_action_prob(board, temp=temp)
            
            features = board.get_features()
            train_examples.append([features, board.current_player, pi])
            
            action_idx = int(np.random.choice(len(pi), p=pi))
            if action_idx == self.board_size * self.board_size:
                action = 'pass'
            else:
                action = (action_idx // self.board_size, action_idx % self.board_size)
                
            board.play_move(action)
            
            if board.is_game_over() or step_count > 120:
                score_est = board.get_ownership_map()
                final_own = np.array(score_est["ownership"], dtype=np.float32)
                lead = score_est["lead"]
                winner = 1 if lead > 0 else (-1 if lead < 0 else 0)
                
                augmented_examples = []
                for x in train_examples:
                    r = 0.0 if winner == 0 else (1.0 if winner == x[1] else -1.0)
                    # 归属图：当前玩家视角 (+1 我方领地, -1 敌方领地)
                    own_current_view = final_own if x[1] == BLACK else -final_own
                    
                    for sym_feat, sym_pi, sym_own in get_symmetries(x[0], x[2], own_current_view, self.board_size):
                        augmented_examples.append([sym_feat, sym_pi, r, sym_own])
                return augmented_examples

    def train_network(self):
        self.network.train()
        
        sample_size = min(len(self.memory), 20000)
        data_pool = random.sample(list(self.memory), sample_size)
        
        for epoch in range(self.epochs):
            total_pi_loss = 0.0
            total_v_loss = 0.0
            total_own_loss = 0.0
            num_batches = 0
            random.shuffle(data_pool)
            
            for i in range(0, len(data_pool), self.batch_size):
                batch = data_pool[i:i+self.batch_size]
                if len(batch) < 16:
                    continue
                
                state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
                pi_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
                v_batch = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(self.device)
                own_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
                
                self.optimizer.zero_grad()
                out_pi, out_v, out_own = self.network(state_batch)
                
                # Multi-task losses
                l_pi = -torch.sum(pi_batch * out_pi) / pi_batch.size(0)
                l_v = F.mse_loss(out_v, v_batch)
                l_own = F.mse_loss(out_own, own_batch)
                
                # Combined Loss with KataGo Ownership weight
                total_loss = l_pi + 1.0 * l_v + 0.5 * l_own
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2.0)
                self.optimizer.step()
                
                total_pi_loss += l_pi.item()
                total_v_loss += l_v.item()
                total_own_loss += l_own.item()
                num_batches += 1
                
            avg_pi = total_pi_loss / max(1, num_batches)
            avg_v = total_v_loss / max(1, num_batches)
            avg_own = total_own_loss / max(1, num_batches)
            print(f"  Epoch {epoch+1}/{self.epochs} | Policy: {avg_pi:.3f} | Value: {avg_v:.3f} | Ownership: {avg_own:.3f} | Total: {avg_pi + avg_v + 0.5*avg_own:.3f}")

    def run(self):
        print(f"[AlphaZero] 17-Channel SE-ResNet Multi-Task Training on device [{self.device}]")
        for i in range(self.iters):
            current_iter = self.start_iteration + i + 1
            print(f"\n--- Iteration {current_iter} (Session: {i+1}/{self.iters}) ---")
            for e in range(self.eps):
                t1 = time.time()
                examples = self.execute_episode()
                self.memory.extend(examples)
                self.total_episodes += 1
                print(f"  Episode {e+1}/{self.eps} (Total: {self.total_episodes}) in {time.time()-t1:.1f}s | Injected: {len(examples)} | Memory: {len(self.memory)}")
            
            self.train_network()
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'iteration': current_iter,
                'total_episodes': self.total_episodes
            }
            torch.save(checkpoint, self.model_path)
            print(f"[AlphaZero] Checkpoint saved: {self.model_path}")

if __name__ == "__main__":
    trainer = AlphaZeroTrainer(board_size=9, iters=100, eps=6, epochs=5, batch_size=128, lr=0.0005)
    trainer.run()
