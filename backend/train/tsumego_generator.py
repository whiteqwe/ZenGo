import random
import numpy as np
from backend.engine.board import GoBoard, BLACK, WHITE, EMPTY

# 经典 9x9 死活、手筋与官子题库大全 (Comprehensive Tsumego & Tesuji Bank)
TSUMEGO_TEMPLATES = [
    # 1. 角部直三 / 弯三做眼与破眼 (Corner 3-space eyespace)
    {
        "name": "角部二之二急所 (2-2 Vital Point)",
        "black": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)],
        "white": [(0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0)],
        "to_play": BLACK
    },
    # 2. 紧气对杀 (Semeai 3-liberty race)
    {
        "name": "角部紧气对杀 (Corner Semeai Race)",
        "black": [(0, 1), (1, 1), (1, 0)],
        "white": [(0, 2), (1, 2), (2, 1), (2, 0)],
        "to_play": BLACK
    },
    # 3. 刀把五 / 丁四要点 (Bent-4 & Vital Eye Shape)
    {
        "name": "边角做眼死穴 (Border Vital Eye)",
        "black": [(0, 1), (0, 2), (0, 3), (1, 1), (1, 3)],
        "white": [(0, 4), (1, 4), (2, 1), (2, 2), (2, 3)],
        "to_play": WHITE
    },
    # 4. 扑与倒脱靴 / 挖断手筋 (Cut & Clamp Tesuji)
    {
        "name": "挖断与扑吃手筋 (Clamp & Atari Tesuji)",
        "black": [(3, 3), (3, 5), (4, 4), (5, 3), (5, 5)],
        "white": [(3, 4), (4, 3), (4, 5), (5, 4)],
        "to_play": BLACK
    },
    # 5. 官子分界与扳粘收束 (Endgame Yose Border Sente)
    {
        "name": "官子一二路先手收束 (Endgame Sente Yose)",
        "black": [(0, 2), (1, 2), (2, 2), (3, 3), (4, 4), (5, 4), (6, 3), (7, 2), (8, 2)],
        "white": [(0, 4), (1, 4), (2, 4), (3, 5), (4, 5), (5, 5), (6, 5), (7, 4), (8, 4)],
        "to_play": BLACK
    },
    # 6. 金鸡独立与双活 (Seki & Double Liberties)
    {
        "name": "公气双活与大龙对杀 (Seki & Liberties)",
        "black": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],
        "white": [(0, 3), (1, 3), (2, 3), (3, 2), (3, 1), (3, 0)],
        "to_play": WHITE
    },
    # 7. 经典板六做活 (Plate-Six Vital Eye)
    {
        "name": "角部板六做活 (Plate-Six Eye)",
        "black": [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
        "white": [(0, 0), (0, 4), (1, 0), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)],
        "to_play": BLACK
    },
    # 8. 倒脱靴吃子手筋 (Under the Stones)
    {
        "name": "倒脱靴手筋 (Under-the-stones Tesuji)",
        "black": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "white": [(0, 2), (1, 2), (2, 0), (2, 1)],
        "to_play": BLACK
    },
    # 9. 大头鬼紧气杀 (Crane's Nest / Semeai Tesuji)
    {
        "name": "大头鬼紧气杀 (Crane Nest Semeai)",
        "black": [(0, 1), (1, 1), (2, 1), (2, 2)],
        "white": [(0, 2), (1, 2), (1, 3), (2, 3), (3, 1), (3, 2)],
        "to_play": BLACK
    },
    # 10. 夹与尖顶官子 (Endgame Clamp & Sente)
    {
        "name": "二路尖顶与夹 (Endgame 2nd-line Clamp)",
        "black": [(1, 1), (2, 1), (3, 1), (4, 2)],
        "white": [(1, 3), (2, 3), (3, 3), (4, 3)],
        "to_play": BLACK
    },
    # 11. 中盘大龙分断与包围 (Midgame Cut & Surround)
    {
        "name": "大龙分断对攻 (Midgame Dragon Cut)",
        "black": [(2, 3), (3, 3), (4, 3), (5, 3), (3, 4)],
        "white": [(2, 5), (3, 5), (4, 5), (5, 5), (4, 4)],
        "to_play": BLACK
    },
    # 12. 劫争做活 (Ko Fight for Life)
    {
        "name": "角部生死劫争 (Corner Ko Fight)",
        "black": [(0, 1), (1, 0), (1, 2), (2, 1)],
        "white": [(0, 2), (1, 3), (2, 2), (3, 1), (2, 0)],
        "to_play": BLACK
    }
]

def generate_tactical_board(board_size=9):
    """
    随机生成一个高强度的局部死活题、紧气对杀或官子残局盘面。
    用于在通宵训练中混合注入，强制学生网络掌握死活基本功与官子精算。
    """
    board = GoBoard(size=board_size)
    template = random.choice(TSUMEGO_TEMPLATES)
    
    # 随机旋转与翻转以扩增死活题方向 (8 种空间对称变化)
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
            
    # 社区死活数据增强技巧：50% 几率在远离死活区域的远端随机散布 1~2 颗背景棋子（模拟真实实战死活局面）
    if random.random() < 0.5:
        for _ in range(random.randint(1, 2)):
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
