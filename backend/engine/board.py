import numpy as np
import copy

EMPTY = 0
BLACK = 1
WHITE = -1

HANDICAP_POINTS_9X9 = {
    2: [(2, 6), (6, 2)],                     # G7, C3
    3: [(2, 6), (6, 2), (6, 6)],             # G7, C3, C7
    4: [(2, 2), (2, 6), (6, 2), (6, 6)],     # G3, G7, C3, C7
    5: [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)] # + Tengen E5
}

HANDICAP_POINTS_13X13 = {
    2: [(3, 9), (9, 3)],                                                 # K10, D4
    3: [(3, 9), (9, 3), (9, 9)],                                         # K10, D4, K4
    4: [(3, 3), (3, 9), (9, 3), (9, 9)],                                 # D10, K10, D4, K4 (4星位)
    5: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)],                         # 4星位 + 天元
    6: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 3), (6, 9)],                 # 4星位 + 2边星
    7: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 3), (6, 9), (6, 6)],         # 6星位 + 天元
    8: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 3), (6, 9), (3, 6), (9, 6)], # 8星位
    9: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 3), (6, 9), (3, 6), (9, 6), (6, 6)] # 8星位 + 天元 (九子)
}

HANDICAP_POINTS_19X19 = {
    2: [(3, 15), (15, 3)],                                                 # Q16, D4
    3: [(3, 15), (15, 3), (15, 15)],                                         # Q16, D4, Q4
    4: [(3, 3), (3, 15), (15, 3), (15, 15)],                                 # D16, Q16, D4, Q4 (4星位)
    5: [(3, 3), (3, 15), (15, 3), (15, 15), (9, 9)],                         # 4星位 + 天元 (K10)
    6: [(3, 3), (3, 15), (15, 3), (15, 15), (9, 3), (9, 15)],                 # 4星位 + 2边星 (D10, Q10)
    7: [(3, 3), (3, 15), (15, 3), (15, 15), (9, 3), (9, 15), (9, 9)],         # 6星位 + 天元
    8: [(3, 3), (3, 15), (15, 3), (15, 15), (9, 3), (9, 15), (3, 9), (15, 9)], # 8星位 (4角 + 4边)
    9: [(3, 3), (3, 15), (15, 3), (15, 15), (9, 3), (9, 15), (3, 9), (15, 9), (9, 9)] # 8星位 + 天元 (九子)
}

class GoBoard:
    def __init__(self, size=9, komi=7.0, handicap=0):
        self.size = size
        self.komi = komi
        self.handicap = handicap
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = BLACK
        self.ko_point = None
        self.last_move = None
        self.passed_consecutively = 0
        self.history = set()
        self.captures = {BLACK: 0, WHITE: 0} # Number of stones captured by BLACK / WHITE
        self.move_history = [] # List of (color, action)
        self.board_states = [] # List of past board snapshots for 17/29-channel features
        self.resigned_player = None
        
        # Setup handicap stones
        if handicap >= 2:
            if size == 19:
                pts_dict = HANDICAP_POINTS_19X19
            elif size == 13:
                pts_dict = HANDICAP_POINTS_13X13
            else:
                pts_dict = HANDICAP_POINTS_9X9
            pts = pts_dict.get(handicap, pts_dict[2])
            for r, c in pts:
                self.board[r, c] = BLACK
                self.move_history.append((BLACK, (r, c)))
            self.current_player = WHITE # White moves first after handicap stones
            self.komi = 0.5            # Standard handicap komi
            
        self.record_history()

    def record_history(self):
        self.history.add(self.board.tobytes())
        if not hasattr(self, 'board_states'):
            self.board_states = []
        self.board_states.append(self.board.copy())

    def copy(self):
        new_board = GoBoard.__new__(GoBoard)
        new_board.size = self.size
        new_board.komi = self.komi
        new_board.handicap = self.handicap
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.last_move = self.last_move
        new_board.passed_consecutively = self.passed_consecutively
        new_board.history = self.history.copy()
        new_board.captures = self.captures.copy()
        new_board.move_history = list(self.move_history)
        new_board.board_states = [b.copy() for b in getattr(self, 'board_states', [self.board])]
        new_board.resigned_player = self.resigned_player
        return new_board

    def copy_fast(self):
        new_board = GoBoard.__new__(GoBoard)
        new_board.size = self.size
        new_board.komi = self.komi
        new_board.handicap = self.handicap
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.last_move = self.last_move
        new_board.passed_consecutively = self.passed_consecutively
        new_board.history = self.history.copy()
        new_board.captures = self.captures.copy()
        new_board.move_history = list(self.move_history)
        new_board.board_states = [b.copy() for b in getattr(self, 'board_states', [self.board])]
        new_board.resigned_player = self.resigned_player
        return new_board

    def is_on_board(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def get_neighbors(self, r, c):
        neighbors = []
        if r > 0: neighbors.append((r-1, c))
        if r < self.size - 1: neighbors.append((r+1, c))
        if c > 0: neighbors.append((r, c-1))
        if c < self.size - 1: neighbors.append((r, c+1))
        return neighbors

    def get_group_and_liberties(self, r, c):
        color = self.board[r, c]
        if color == EMPTY:
            return set(), set()
        
        group = set()
        liberties = set()
        frontier = [(r, c)]
        
        while frontier:
            curr_r, curr_c = frontier.pop()
            if (curr_r, curr_c) in group:
                continue
            group.add((curr_r, curr_c))
            
            for nr, nc in self.get_neighbors(curr_r, curr_c):
                n_color = self.board[nr, nc]
                if n_color == EMPTY:
                    liberties.add((nr, nc))
                elif n_color == color and (nr, nc) not in group:
                    frontier.append((nr, nc))
                    
        return group, liberties

    def is_valid_move(self, action):
        if action == 'pass':
            return True
            
        r, c = action
        if not self.is_on_board(r, c):
            return False
        if self.board[r, c] != EMPTY:
            return False
        if (r, c) == self.ko_point:
            return False
            
        board_copy = np.copy(self.board)
        self.board[r, c] = self.current_player
        
        captured_stones = set()
        opponent = -self.current_player
        for nr, nc in self.get_neighbors(r, c):
            if self.board[nr, nc] == opponent:
                grp, libs = self.get_group_and_liberties(nr, nc)
                if len(libs) == 0:
                    captured_stones.update(grp)
                    
        for cr, cc in captured_stones:
            self.board[cr, cc] = EMPTY
            
        _, my_libs = self.get_group_and_liberties(r, c)
        
        is_superko = self.board.tobytes() in self.history
        
        self.board = board_copy
        
        if len(my_libs) == 0 and len(captured_stones) == 0:
            return False
            
        if is_superko:
            return False
            
        return True

    def play_move(self, action):
        if not self.is_valid_move(action):
            raise ValueError(f"Invalid move: {action}")
            
        color = self.current_player
        self.move_history.append((color, action))

        if action == 'pass':
            self.passed_consecutively += 1
            self.current_player = -self.current_player
            self.ko_point = None
            self.last_move = 'pass'
            return

        r, c = action
        self.passed_consecutively = 0
        self.board[r, c] = color
        
        captured_stones = set()
        opponent = -color
        for nr, nc in self.get_neighbors(r, c):
            if self.board[nr, nc] == opponent:
                grp, libs = self.get_group_and_liberties(nr, nc)
                if len(libs) == 0:
                    captured_stones.update(grp)
                    
        for cr, cc in captured_stones:
            self.board[cr, cc] = EMPTY
            
        self.captures[color] += len(captured_stones)

        if len(captured_stones) == 1:
            _, my_libs = self.get_group_and_liberties(r, c)
            if len(my_libs) == 1 and len(self.get_group_and_liberties(r, c)[0]) == 1:
                self.ko_point = list(captured_stones)[0]
            else:
                self.ko_point = None
        else:
            self.ko_point = None
            
        self.current_player = opponent
        self.last_move = action
        self.record_history()

    def resign(self, player=None):
        """Player resigns the game. If player is None, current_player resigns."""
        if player is None:
            player = self.current_player
        self.resigned_player = player

    def is_game_over(self):
        return self.resigned_player is not None or self.passed_consecutively >= 2

    def get_legal_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_valid_move((r, c)):
                    moves.append((r, c))
        moves.append('pass')
        return moves
        
    def get_features(self, history_steps=8):
        """
        29-channel SOTA Tactical & History representation (AlphaGo / KataGo SOTA standard):
        - Channels 0..7: Current player stones across past 8 steps (T, T-1, ..., T-7)
        - Channels 8..15: Opponent stones across past 8 steps (T, T-1, ..., T-7)
        - Channel 16: Current player stones with 1 liberty (ATARI / 叫吃 / 危险状态)
        - Channel 17: Current player stones with 2 liberties (气紧状态)
        - Channel 18: Current player stones with 3 liberties
        - Channel 19: Current player stones with >= 4 liberties (安全厚实)
        - Channel 20: Opponent stones with 1 liberty (对方被叫吃 / 可提子目标)
        - Channel 21: Opponent stones with 2 liberties (对方气紧 / 可紧气追杀)
        - Channel 22: Opponent stones with 3 liberties
        - Channel 23: Opponent stones with >= 4 liberties
        - Channel 24: Direct Capture Points for Current Player (落子可提吃对方)
        - Channel 25: Direct Capture Points for Opponent (对方落子可提吃我方)
        - Channel 26: Self-Atari Points for Current Player (我方自杀送吃点)
        - Channel 27: Self-Atari Points for Opponent (对方自杀送吃点)
        - Channel 28: Turn indicator (1.0 for Black, 0.0 for White)
        """
        num_channels = 29
        features = np.zeros((num_channels, self.size, self.size), dtype=np.float32)
        
        # 1. Past 8 History snapshots
        boards = []
        if hasattr(self, 'board_states') and len(self.board_states) > 0:
            boards = self.board_states
        else:
            boards = [self.board.copy()]

        for step in range(history_steps):
            if step < len(boards):
                past_b = boards[-(step + 1)]
            else:
                past_b = np.zeros((self.size, self.size), dtype=int)
                
            features[step] = (past_b == self.current_player).astype(np.float32)
            features[history_steps + step] = (past_b == -self.current_player).astype(np.float32)

        # 2. Tactical Liberty Planes (8 channels: 16..23)
        visited = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != EMPTY and (r, c) not in visited:
                    grp, libs = self.get_group_and_liberties(r, c)
                    visited.update(grp)
                    color = self.board[r, c]
                    num_libs = len(libs)
                    
                    base_ch = 16 if color == self.current_player else 20
                    if num_libs == 1:
                        ch = base_ch + 0 # Atari (1-liberty / 叫吃)
                    elif num_libs == 2:
                        ch = base_ch + 1 # 2 liberties (气紧)
                    elif num_libs == 3:
                        ch = base_ch + 2 # 3 liberties
                    else:
                        ch = base_ch + 3 # >= 4 liberties (安全)
                        
                    for gr, gc in grp:
                        features[ch, gr, gc] = 1.0

        # 3. Direct Capture & Self-Atari Tactical Foresight (4 channels: 24..27)
        curr_p = self.current_player
        opp_p = -curr_p
        
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == EMPTY:
                    # Test current player move at (r, c)
                    self.board[r, c] = curr_p
                    curr_captures = any(
                        self.is_on_board(nr, nc) and self.board[nr, nc] == opp_p and 
                        len(self.get_group_and_liberties(nr, nc)[1]) == 0
                        for nr, nc in self.get_neighbors(r, c)
                    )
                    _, curr_libs = self.get_group_and_liberties(r, c)
                    if curr_captures:
                        features[24, r, c] = 1.0
                    elif len(curr_libs) == 1:
                        features[26, r, c] = 1.0
                        
                    # Test opponent move at (r, c)
                    self.board[r, c] = opp_p
                    opp_captures = any(
                        self.is_on_board(nr, nc) and self.board[nr, nc] == curr_p and 
                        len(self.get_group_and_liberties(nr, nc)[1]) == 0
                        for nr, nc in self.get_neighbors(r, c)
                    )
                    _, opp_libs = self.get_group_and_liberties(r, c)
                    if opp_captures:
                        features[25, r, c] = 1.0
                    elif len(opp_libs) == 1:
                        features[27, r, c] = 1.0
                        
                    self.board[r, c] = EMPTY # Reset

        # 4. Turn indicator (channel 28)
        features[28] = np.full((self.size, self.size), 1.0 if self.current_player == BLACK else 0.0, dtype=np.float32)
        return features

    def get_ownership_map(self):
        """
        Calculates a 9x9 ownership map for territory visualization and scoring:
        +1.0: Black alive stones and surrounded Black territory
        -1.0: White alive stones and surrounded White territory
         0.0: Neutral / Dame / Disputed points
        """
        ownership = np.zeros((self.size, self.size), dtype=np.float32)
        visited = set()
        black_territory = 0
        white_territory = 0
        
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == BLACK:
                    ownership[r, c] = 1.0
                    black_territory += 1
                elif self.board[r, c] == WHITE:
                    ownership[r, c] = -1.0
                    white_territory += 1
                elif (r, c) not in visited:
                    empty_group = set()
                    borders = set()
                    frontier = [(r, c)]
                    
                    while frontier:
                        curr_r, curr_c = frontier.pop()
                        if (curr_r, curr_c) in empty_group:
                            continue
                        empty_group.add((curr_r, curr_c))
                        visited.add((curr_r, curr_c))
                        
                        for nr, nc in self.get_neighbors(curr_r, curr_c):
                            if self.board[nr, nc] == EMPTY and (nr, nc) not in empty_group:
                                frontier.append((nr, nc))
                            elif self.board[nr, nc] != EMPTY:
                                borders.add(self.board[nr, nc])
                                
                    if len(borders) == 1:
                        border_color = list(borders)[0]
                        if border_color == BLACK:
                            for er, ec in empty_group:
                                ownership[er, ec] = 1.0
                            black_territory += len(empty_group)
                        else:
                            for er, ec in empty_group:
                                ownership[er, ec] = -1.0
                            white_territory += len(empty_group)
                            
        komi = getattr(self, 'komi', 7.0)
        white_score = white_territory + komi
        black_score = black_territory
        lead = black_score - white_score
        
        return {
            "ownership": ownership.tolist(),
            "black_territory": black_territory,
            "white_territory": white_territory,
            "komi": komi,
            "white_total": round(white_score, 1),
            "black_total": round(black_score, 1),
            "lead": round(lead, 1),
            "lead_text": f"黑领先 {abs(lead):.1f}目" if lead >= 0 else f"白领先 {abs(lead):.1f}目"
        }

    def compute_score(self):
        """Area scoring (Tromp-Taylor rules) or resignation score."""
        if self.resigned_player == BLACK:
            return -999.0 # White wins by resignation
        elif self.resigned_player == WHITE:
            return 999.0 # Black wins by resignation

        est = self.get_ownership_map()
        return est["lead"]
