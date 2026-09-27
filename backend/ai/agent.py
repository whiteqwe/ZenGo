import os
import numpy as np
import torch
from backend.ai.network import AlphaGoZeroNet
from backend.ai.mcts_alpha import MCTSAlpha, index_to_action
from backend.engine.board import BLACK, WHITE

def get_safe_device():
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1, 1, device='cuda')
            _ = x + 1
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")

class GoAgent:
    def __init__(self, board_size=9, model_path=None, num_simulations=160):
        self.device = get_safe_device()
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.network = AlphaGoZeroNet(board_size=board_size, in_channels=29, num_filters=128, num_res_blocks=8).to(self.device)
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(os.path.dirname(__file__), 'best_model_13x13.pth' if board_size == 13 else 'best_model.pth')
        self.last_mtime = 0
        self.load_model(force=True)
        self.mcts = MCTSAlpha(self.network, num_simulations=num_simulations, batch_size=8, device=self.device)

    def set_simulations(self, num_simulations):
        self.num_simulations = num_simulations
        if self.mcts:
            self.mcts.num_simulations = num_simulations

    def set_board_size(self, board_size):
        if self.board_size != board_size:
            self.board_size = board_size
            self.network = AlphaGoZeroNet(board_size=board_size, in_channels=29, num_filters=128, num_res_blocks=8).to(self.device)
            self.model_path = os.path.join(os.path.dirname(__file__), 'best_model_13x13.pth' if board_size == 13 else 'best_model.pth')
            self.last_mtime = 0
            self.load_model(force=True)
            self.mcts = MCTSAlpha(self.network, num_simulations=self.num_simulations, batch_size=8, device=self.device)

    def load_model(self, force=False):
        if os.path.exists(self.model_path):
            try:
                mtime = os.path.getmtime(self.model_path)
                if force or mtime > self.last_mtime:
                    ckpt = torch.load(self.model_path, map_location=self.device)
                    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                        self.network.load_state_dict(ckpt['model_state_dict'])
                        total_ep = ckpt.get('total_episodes', 0)
                        print(f"[GoAgent] Hot-reloaded latest model (Total games: {total_ep})")
                    else:
                        self.network.load_state_dict(ckpt)
                        print(f"[GoAgent] Hot-reloaded latest model weights.")
                    self.last_mtime = mtime
            except Exception as e:
                self.network = AlphaGoZeroNet(board_size=self.board_size, in_channels=29, num_filters=128, num_res_blocks=8).to(self.device)
                self.last_mtime = mtime
                print(f"[GoAgent] Upgraded architecture: Initialized fresh 29-channel SOTA 4-Head SE-ResNet weights.")
        else:
            if force:
                print("[GoAgent] Initialized fresh 29-channel SOTA 4-Head SE-ResNet weights.")
        self.network.eval()

    def get_move(self, board, temp=0.0):
        """Returns the chosen action (r, c) or 'pass' using MCTS."""
        self.load_model() # 每次落子前自动检测并热加载最新训练的模型
        action_probs, root_val, candidates = self.mcts.get_action_prob(board, temp=temp, return_search_info=True)
        if temp == 0:
            action_idx = int(np.argmax(action_probs))
        else:
            action_idx = int(np.random.choice(len(action_probs), p=action_probs))
            
        return index_to_action(action_idx, board.size)

    def get_evaluation(self, board):
        """Returns objective MCTS-searched winrate & Score Lead analysis (KataGo / Leela Zero standard)."""
        self.load_model()
        try:
            _, root_val, candidates = self.mcts.get_action_prob(board, temp=0.0, return_search_info=True)
            
            # root_val 是当前行动方的期望价值
            if board.current_player == BLACK:
                black_winrate = (root_val + 1.0) / 2.0
            else:
                black_winrate = 1.0 - (root_val + 1.0) / 2.0
                
            black_winrate = max(0.01, min(0.99, black_winrate))
            
            # 4-Head 领地与目数差精准前向推断
            res = self.network.predict(board.get_features(), device=self.device)
            raw_lead = res[3]
            score_lead = round(float(raw_lead if board.current_player == BLACK else -raw_lead), 1)
            
            if score_lead > 0.3:
                lead_text = f"黑领先 {score_lead:.1f} 目"
            elif score_lead < -0.3:
                lead_text = f"白领先 {abs(score_lead):.1f} 目"
            else:
                lead_text = "势均力敌 (0.0目)"

            return {
                "black_winrate": round(float(black_winrate) * 100, 1),
                "white_winrate": round(float(1.0 - black_winrate) * 100, 1),
                "score_lead": score_lead,
                "lead_text": lead_text,
                "top_candidates": candidates[:5]
            }
        except Exception as e:
            return {
                "black_winrate": 50.0,
                "white_winrate": 50.0,
                "value": 0.0,
                "top_candidates": []
            }
