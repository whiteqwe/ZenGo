import random
import numpy as np
from backend.engine.board import GoBoard, BLACK, WHITE, EMPTY

# 13x13 经典死活、定式开局、手筋与官子题库模版 (13x13 Tsumego & Joseki Templates)
TSUMEGO_13X13_TEMPLATES = [
    # 1. 13x13 经典星位与点三三定式 (Star Point & 3-3 Invasion)
    {
        "name": "星位点三三与外势转换 (3-3 Invasion & Moyo)",
        "black": [(3, 3), (2, 2), (2, 3), (3, 2), (1, 3)],
        "white": [(3, 4), (4, 3), (4, 4), (4, 2), (2, 4)],
        "to_play": BLACK
    },
    # 2. 13x13 小飞挂角与夹攻 (Knight Approach & Pincer)
    {
        "name": "小飞挂角与一间低夹 (Knight Approach & Pincer)",
        "black": [(3, 3), (3, 5), (5, 3)],
        "white": [(2, 4), (3, 4)],
        "to_play": WHITE
    },
    # 3. 13x13 角部大龙死活急所 (Corner Dragon Life & Death 2-2)
    {
        "name": "角部二之二做活急所 (Corner 2-2 Vital)",
        "black": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0)],
        "white": [(0, 4), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0)],
        "to_play": BLACK
    },
    # 4. 13x13 紧气对杀 (Semeai Race)
    {
        "name": "边角紧气对杀 (Border Semeai Race)",
        "black": [(1, 1), (1, 2), (2, 1)],
        "white": [(1, 3), (2, 3), (3, 2), (3, 1)],
        "to_play": BLACK
    },
    # 5. 13x13 中腹天元分界与大模样 (Tengen Center Moyo Battle)
    {
        "name": "中腹天元大模样分界 (Center Moyo Division)",
        "black": [(6, 4), (6, 5), (6, 6), (7, 6), (8, 6)],
        "white": [(6, 7), (6, 8), (5, 6), (4, 6), (5, 7)],
        "to_play": BLACK
    },
    # 6. 13x13 官子先手一二路扳粘 (Endgame Yose 1st/2nd Line Sente)
    {
        "name": "官子一二路先手收束 (Endgame 13x13 Sente Yose)",
        "black": [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 5), (6, 6), (7, 5), (8, 4), (9, 3)],
        "white": [(0, 5), (1, 5), (2, 5), (3, 6), (4, 7), (5, 7), (6, 8), (7, 7), (8, 6), (9, 5)],
        "to_play": BLACK
    },
    # 7. 13x13 倒脱靴与扑吃手筋 (Under the Stones & Clamp)
    {
        "name": "倒脱靴与扑劫手筋 (Under-the-stones Tesuji)",
        "black": [(1, 1), (1, 2), (2, 1), (2, 2)],
        "white": [(1, 3), (2, 3), (3, 1), (3, 2), (0, 2)],
        "to_play": BLACK
    },
    # 8. 13x13 经典板六做活 (Plate-Six Eye 13x13)
    {
        "name": "角部板六做活 (Plate-Six Vital Eye)",
        "black": [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
        "white": [(0, 0), (0, 4), (1, 0), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)],
        "to_play": BLACK
    }
]

def generate_tactical_board_13x13(board_size=13):
    """
    随机生成 13x13 高强度死活、定式、中腹对攻或官子残局盘面。
    支持 D4 空间 8 种对称变换与实战背景杂音扰动。
    """
    board = GoBoard(size=board_size)
    template = random.choice(TSUMEGO_13X13_TEMPLATES)
    
    rot = random.randint(0, 3)
    flip = random.choice([True, False])
    
    def transform(r, c):
        for _ in range(rot):
            r, c = c, board_size - 1 - r
        if flip:
            c = board_size - 1 - c
        return (r, c)
        
    col_letters = "ABCDEFGHJKLMNOPQRST"
    initial_stones = []

    for r, c in template["black"]:
        tr, tc = transform(r, c)
        if board.is_on_board(tr, tc):
            board.board[tr, tc] = BLACK
            c_char = col_letters[tc]
            row_num = board_size - tr
            initial_stones.append(["B", f"{c_char}{row_num}"])
            
    for r, c in template["white"]:
        tr, tc = transform(r, c)
        if board.is_on_board(tr, tc):
            board.board[tr, tc] = WHITE
            c_char = col_letters[tc]
            row_num = board_size - tr
            initial_stones.append(["W", f"{c_char}{row_num}"])
            
    # 50% 几率在远端空旷处随机散落 1~3 颗背景子，模拟真实 13x13 实战
    if random.random() < 0.5:
        for _ in range(random.randint(1, 3)):
            noise_r = random.randint(0, board_size - 1)
            noise_c = random.randint(0, board_size - 1)
            if board.board[noise_r, noise_c] == EMPTY:
                color_val = random.choice([BLACK, WHITE])
                tag = "B" if color_val == BLACK else "W"
                board.board[noise_r, noise_c] = color_val
                c_char = col_letters[noise_c]
                row_num = board_size - noise_r
                initial_stones.append([tag, f"{c_char}{row_num}"])
            
    board.current_player = template["to_play"]
    board.initial_stones = initial_stones
    board.move_history = []
    board.record_history()
    return board
