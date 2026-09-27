import numpy as np
import math
import copy

class MCTSNode:
    def __init__(self, parent=None, prior_p=1.0, action=None):
        self.parent = parent
        self.children = {} # action: Node
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior_p
        self.action = action

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=prob, action=action)

    def is_expanded(self):
        return len(self.children) > 0

    def update(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N

    def get_value(self, c_puct, fpu_reduction=0.20):
        # self.Q is the action-value from the parent's perspective
        # FPU (First Play Urgency - KataGo / Leela Zero standard):
        # 未访问节点 (N=0) 动态继承父节点 Q 值并略作折减，避免在劣势分支上漫无目的地浪费探索次数
        if self.N == 0:
            q_val = (self.parent.Q - fpu_reduction) if (self.parent and self.parent.N > 0) else 0.0
        else:
            q_val = self.Q
            
        u = c_puct * self.P * math.sqrt(max(1, self.parent.N if self.parent else 1)) / (1 + self.N)
        return q_val + u

def action_to_index(action, size):
    if action == 'pass':
        return size * size
    r, c = action
    return r * size + c

def index_to_action(idx, size):
    if idx == size * size:
        return 'pass'
    return (idx // size, idx % size)

class MCTSAlpha:
    def __init__(self, network, c_puct=1.5, num_simulations=400, batch_size=8, device="cpu"):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.device = device
        self.eval_cache = {}

    def get_action_prob(self, state, temp=1.0, add_noise=None, return_search_info=False):
        import torch
        root = MCTSNode()
        if add_noise is None:
            add_noise = (temp > 0.0)
            
        sims_done = 0
        while sims_done < self.num_simulations:
            batch_leaves = []
            cur_batch = min(self.batch_size, self.num_simulations - sims_done)
            
            for _ in range(cur_batch):
                node = root
                board_copy = state.copy_fast()
                path = [node]
                
                # Selection with Virtual Loss
                while node.is_expanded():
                    best_action = None
                    best_value = -float('inf')
                    for action, child in node.children.items():
                        val = child.get_value(self.c_puct)
                        if val > best_value:
                            best_value = val
                            best_action = action
                    node = node.children[best_action]
                    board_copy.play_move(best_action)
                    path.append(node)
                    
                    # Virtual loss to explore distinct branches within the parallel mini-batch
                    node.N += 1
                    node.W -= 1.0
                    node.Q = node.W / node.N

                batch_leaves.append((node, board_copy, path))

            # Filter leaves that require GPU neural network inference
            uncached_leaves = []
            features_list = []
            
            for i, (node, board_copy, path) in enumerate(batch_leaves):
                if not board_copy.is_game_over():
                    key = (board_copy.board.tobytes(), board_copy.current_player)
                    if key not in self.eval_cache:
                        uncached_leaves.append(i)
                        features_list.append(board_copy.get_features())

            # Batched GPU Tensor Inference (single forward pass for entire batch)
            if len(features_list) > 0:
                batch_tensor = torch.from_numpy(np.stack(features_list)).to(device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    log_policies, vals, _, _ = self.network(batch_tensor)
                    policies = torch.exp(log_policies).cpu().numpy()
                    values = vals.cpu().numpy().flatten()
                
                for f_idx, leaf_idx in enumerate(uncached_leaves):
                    _, b_copy, _ = batch_leaves[leaf_idx]
                    key = (b_copy.board.tobytes(), b_copy.current_player)
                    p_raw = policies[f_idx]
                    v_raw = float(values[f_idx])
                    
                    legal_moves = b_copy.get_legal_moves()
                    action_probs = []
                    prob_sum = 0
                    for action in legal_moves:
                        idx = action_to_index(action, b_copy.size)
                        prob = float(p_raw[idx])
                        action_probs.append((action, prob))
                        prob_sum += prob
                    if prob_sum > 0:
                        norm_probs = [(a, p / prob_sum) for a, p in action_probs]
                    else:
                        norm_probs = [(a, 1.0 / len(legal_moves)) for a in legal_moves]
                    self.eval_cache[key] = (norm_probs, v_raw)

            # Node Expansion and Backpropagation
            for i, (node, board_copy, path) in enumerate(batch_leaves):
                # Revert virtual loss on traversed path
                for p_node in path[1:]:
                    p_node.N -= 1
                    p_node.W += 1.0
                    p_node.Q = p_node.W / p_node.N if p_node.N > 0 else 0.0

                if not board_copy.is_game_over():
                    key = (board_copy.board.tobytes(), board_copy.current_player)
                    action_probs, value = self.eval_cache[key]
                    
                    if node == root and add_noise and len(action_probs) > 0:
                        noise = np.random.dirichlet([0.3] * len(action_probs))
                        root_probs = [(a, 0.75 * p + 0.25 * n) for (a, p), n in zip(action_probs, noise)]
                        node.expand(root_probs)
                    else:
                        node.expand(action_probs)
                else:
                    score = board_copy.compute_score()
                    if score > 0:
                        value = 1.0 if board_copy.current_player == 1 else -1.0
                    elif score < 0:
                        value = -1.0 if board_copy.current_player == 1 else 1.0
                    else:
                        value = 0.0

                # Backpropagation
                curr_v = -value
                curr_node = node
                while curr_node is not None:
                    curr_node.update(curr_v)
                    curr_v = -curr_v
                    curr_node = curr_node.parent

            sims_done += cur_batch

        if len(root.children) == 0:
            full_probs = np.zeros(state.size * state.size + 1)
            full_probs[-1] = 1.0 # Pass
            return (full_probs, 0.0, []) if return_search_info else full_probs

        action_data = []
        for action, child in root.children.items():
            our_winrate = (child.Q + 1.0) / 2.0
            our_winrate = max(0.01, min(0.99, our_winrate))
            action_data.append((action, child.N, child.P, our_winrate, child.Q))

        actions = [x[0] for x in action_data]
        visits = [x[1] for x in action_data]
        
        full_probs = np.zeros(state.size * state.size + 1)
        if temp == 0:
            best_idx = int(np.argmax(visits))
            best_action = actions[best_idx]
            full_probs[action_to_index(best_action, state.size)] = 1.0
        else:
            v_arr = np.array(visits, dtype=np.float64) ** (1.0 / max(0.01, temp))
            v_sum = np.sum(v_arr)
            probs = v_arr / v_sum if v_sum > 0 else np.ones(len(visits)) / len(visits)
            for action, prob in zip(actions, probs):
                idx = action_to_index(action, state.size)
                full_probs[idx] = prob
                
        if return_search_info:
            sorted_candidates = sorted(action_data, key=lambda x: x[1], reverse=True)
            candidate_list = []
            for a, n, p, w, q in sorted_candidates:
                candidate_list.append({
                    "move": a,
                    "prob": round(float(p) * 100, 1),
                    "winrate": round(float(w) * 100, 1),
                    "visits": int(n)
                })
            root_val = float(sorted_candidates[0][4]) if sorted_candidates else 0.0
            return full_probs, root_val, candidate_list

        return full_probs
