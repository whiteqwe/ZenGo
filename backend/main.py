from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

from backend.engine.board import GoBoard, BLACK, WHITE, EMPTY
from backend.ai.agent import GoAgent
from backend.ai.katago_service import katago_service

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import random
import time

# Global state and settings
current_board_size = 9      # 9 or 13
current_komi = 7.0          # 9x9 Golden Fair Komi (7.0), 13x13 standard (6.5 or 7.0)
current_handicap = 0        # 0 (even), 1 (let_first), 2 (handicap 2), 3 (handicap 3), 4 (handicap 4)
color_mode = "black"        # "black", "white", "random" (猜先)
player_color = BLACK        # Human player color: BLACK (1) or WHITE (-1)

game_board = GoBoard(size=current_board_size, komi=current_komi, handicap=current_handicap)
ai_agent = GoAgent(board_size=current_board_size, num_simulations=60)

opponent_engine = "katago" # "katago" or "alphazero"
referee_engine = "katago"  # "katago" or "alphazero"
resign_mode = "smart"       # "smart", "never", "quick"
current_ai_visits = -1      # -1 (Auto 智能自适应), 150 (Blitz), 500, 1500, 3000, 6000
ai_winrate_history = []
last_evaluation = None
last_dynamic_info = None

class MoveRequest(BaseModel):
    r: int = -1
    c: int = -1
    is_pass: bool = False

class ConfigRequest(BaseModel):
    board_size: Optional[int] = None
    resign_mode: Optional[str] = None
    opponent_engine: Optional[str] = None
    referee_engine: Optional[str] = None
    komi: Optional[float] = None
    color_mode: Optional[str] = None
    handicap: Optional[int] = None
    ai_visits: Optional[int] = None

def compute_dynamic_visits(board_obj, max_ceiling=6000, normal_budget=1600, min_floor=350):
    """
    Dynamically determine the optimal MCTS visit count:
    - Trivial / Opening moves: 350 ~ 500 visits (~0.2s - 0.4s)
    - Normal middle-game positional moves: 1400 ~ 2000 visits (~1.2s - 1.8s, < 3s)
    - Critical dragon life-death, semeai, ko fight: 4500 ~ 6000 visits (~3.5s - 5.5s, < 7s)
    """
    move_num = len(board_obj.move_history)
    size = board_obj.size

    # 1. Very early opening moves (moves 0 - 3): standard corner/tengen points
    if move_num <= 3:
        return min_floor, "⚡ 开局定式 (350次秒回)"

    # 2. Tactical scanning: check stones in Atari (1-2 liberties) & large groups in danger
    atari_stones = 0
    large_group_in_danger = False
    total_stones = 0
    
    visited = set()
    for r in range(size):
        for c in range(size):
            if board_obj.board[r, c] != EMPTY:
                total_stones += 1
                if (r, c) not in visited:
                    grp, libs = board_obj.get_group_and_liberties(r, c)
                    visited.update(grp)
                    if len(libs) == 1:
                        atari_stones += len(grp)
                        if len(grp) >= 2:
                            large_group_in_danger = True
                    elif len(libs) == 2:
                        if len(grp) >= 3:
                            large_group_in_danger = True

    # 3. Decision Matrix
    # Case A: Critical dragon life-and-death or multi-stone atari / semeai
    if large_group_in_danger or atari_stones >= 3:
        return max_ceiling, "🌌 复杂死活绞杀 (6000次神算)"

    # Case B: Local contact fighting (stones in atari / tight liberties)
    if atari_stones >= 1:
        visits = min(max_ceiling, int(normal_budget * 2.2)) # ~3500 visits
        return visits, "👑 局部接触战劫争 (3500次深算)"

    # Case C: Active Middle game
    if 4 <= move_num <= int(size * size * 0.55):
        return normal_budget, "🏆 中盘大局选点 (1600次精算)"

    # Case D: Settled endgame / peaceful territory closure
    if total_stones > int(size * size * 0.70):
        return min_floor + 150, "🏁 终局官子平稳收官 (500次快下)"

    return normal_budget, "🏃 常规平稳选点 (1600次精算)"

def compute_evaluation(force_recompute=False, visits=None):
    global last_evaluation, last_dynamic_info
    if not force_recompute and last_evaluation is not None:
        return last_evaluation
    
    if visits is not None:
        use_visits = visits
    elif current_ai_visits in [-1, "auto"]:
        use_visits, tag = compute_dynamic_visits(game_board)
        last_dynamic_info = {"visits": use_visits, "tag": tag}
    else:
        use_visits = int(current_ai_visits)
        last_dynamic_info = {"visits": use_visits, "tag": f"固定推演 ({use_visits}次)"}

    eval_data = None
    if referee_engine == "katago" and katago_service.is_available():
        try:
            eval_data = katago_service.analyze_board(game_board.move_history, board_size=game_board.size, komi=game_board.komi, max_visits=use_visits)
        except Exception as e:
            print(f"[Main] KataGo analysis error: {e}")
            eval_data = None

    if eval_data is None:
        eval_data = ai_agent.get_evaluation(game_board)

    if eval_data and last_dynamic_info:
        eval_data["dynamic_info"] = last_dynamic_info

    last_evaluation = eval_data
    return last_evaluation

@app.get("/state")
def get_state(recompute_eval: bool = False):
    global last_evaluation
    eval_data = compute_evaluation(force_recompute=recompute_eval) if recompute_eval else last_evaluation

    if eval_data and eval_data.get("ownership"):
        territory_data = {
            "ownership": eval_data["ownership"],
            "black_territory": eval_data.get("black_territory", 0),
            "white_territory": eval_data.get("white_territory", 0),
            "komi": float(game_board.komi),
            "white_total": round(eval_data.get("white_territory", 0) + float(game_board.komi), 1),
            "black_total": round(float(eval_data.get("black_territory", 0)), 1),
            "lead": eval_data.get("fox_lead", eval_data.get("score_lead", 0.0)),
            "lead_text": eval_data.get("fox_lead_text", eval_data.get("lead_text", "势均力敌"))
        }
    else:
        territory_data = game_board.get_ownership_map()

    if eval_data:
        if "score_lead" not in eval_data or eval_data["score_lead"] is None:
            eval_data["score_lead"] = float(territory_data.get("lead", 0.0)) if territory_data else 0.0
        if "lead_text" not in eval_data or not eval_data["lead_text"]:
            lead_val = eval_data["score_lead"]
            eval_data["lead_text"] = f"黑领先 {lead_val:.1f}目" if lead_val >= 0 else f"白领先 {abs(lead_val):.1f}目"

    # Final score calculation (申请数子 / 终局官方裁决：保留高精度 KataGo MCTS 目数与中国规则算力)
    final_score = None
    if game_board.is_game_over():
        if game_board.resigned_player == BLACK:
            final_score = -999.0
        elif game_board.resigned_player == WHITE:
            final_score = 999.0
        elif eval_data and eval_data.get("score_lead") is not None:
            final_score = float(eval_data["score_lead"])
        elif territory_data:
            final_score = float(territory_data.get("lead", 0.0))
        else:
            final_score = float(game_board.compute_score())

    ai_suggests_scoring = bool(
        not game_board.is_game_over()
        and game_board.passed_consecutively == 1
        and game_board.last_move == "pass"
        and len(game_board.move_history) >= 16
    )

    return {
        "board_size": int(game_board.size),
        "board": game_board.board.tolist(),
        "current_player": int(game_board.current_player),
        "is_game_over": bool(game_board.is_game_over()),
        "last_move": game_board.last_move,
        "score": final_score,
        "captures": {
            "black": int(game_board.captures[BLACK]),
            "white": int(game_board.captures[WHITE])
        },
        "move_count": len(game_board.move_history),
        "move_history": game_board.move_history,
        "resigned_player": game_board.resigned_player,
        "opponent_engine": opponent_engine,
        "referee_engine": referee_engine,
        "resign_mode": resign_mode,
        "komi": float(game_board.komi),
        "color_mode": color_mode,
        "player_color": int(player_color),
        "handicap": current_handicap,
        "ai_visits": current_ai_visits,
        "evaluation": eval_data,
        "territory": territory_data,
        "ai_suggests_scoring": ai_suggests_scoring
    }

@app.post("/evaluate")
def evaluate_board():
    """独立局势分析裁判接口：后台异步计算胜率与形势，不阻塞落子"""
    return get_state(recompute_eval=True)

@app.post("/request_scoring")
def request_scoring():
    """Request referee area scoring to settle game immediately and determine winner."""
    if not game_board.is_game_over():
        game_board.passed_consecutively = 2
    return get_state(recompute_eval=True)

@app.get("/score_estimate")
def get_score_estimate():
    state = get_state(recompute_eval=True)
    return state["territory"]

@app.post("/config")
def set_config(req: ConfigRequest):
    global resign_mode, opponent_engine, referee_engine, current_komi, color_mode, current_handicap, current_board_size, current_ai_visits, last_evaluation
    board_changed = False
    if req.board_size in [9, 13, 19]:
        if current_board_size != req.board_size:
            current_board_size = req.board_size
            if current_board_size in [9, 13]:
                ai_agent.set_board_size(current_board_size)
            else:
                opponent_engine = "katago"
                referee_engine = "katago"
            board_changed = True
    if req.resign_mode in ["smart", "never", "quick"]:
        resign_mode = req.resign_mode
    if req.ai_visits is not None and req.ai_visits in [-1, 150, 500, 1500, 3000, 6000, 10000]:
        current_ai_visits = req.ai_visits
        if current_ai_visits != -1:
            ai_agent.set_simulations(min(current_ai_visits, 800))
    if current_board_size == 19:
        opponent_engine = "katago"
        referee_engine = "katago"
    else:
        if req.opponent_engine in ["katago", "alphazero"]:
            opponent_engine = req.opponent_engine
        if req.referee_engine in ["katago", "alphazero"]:
            referee_engine = req.referee_engine
    if req.komi is not None:
        current_komi = float(req.komi)
        game_board.komi = current_komi
    if req.color_mode in ["black", "white", "random"]:
        color_mode = req.color_mode
    if req.handicap is not None:
        current_handicap = int(req.handicap)

    if board_changed:
        return reset_game()

    return {
        "board_size": current_board_size,
        "resign_mode": resign_mode,
        "opponent_engine": opponent_engine,
        "referee_engine": referee_engine,
        "komi": current_komi,
        "color_mode": color_mode,
        "handicap": current_handicap,
        "player_color": int(player_color),
        "ai_visits": current_ai_visits
    }

@app.post("/move")
def play_move(req: MoveRequest):
    global last_evaluation
    if game_board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")
        
    action = 'pass' if req.is_pass else (req.r, req.c)
    if not game_board.is_valid_move(action):
        raise HTTPException(status_code=400, detail="Invalid move")
        
    game_board.play_move(action)
    # ⚡ 极速秒回！落子操作 0ms 阻塞，不强行插入大算力裁判分析
    return get_state(recompute_eval=False)

@app.post("/ai_move")
def ai_move():
    global ai_winrate_history, last_evaluation, last_dynamic_info
    if game_board.is_game_over():
        return get_state(recompute_eval=False)

    # 1. 动态算力调度（智能自适应 vs 用户手动指定）
    if current_ai_visits in [-1, "auto"]:
        use_visits, tag = compute_dynamic_visits(game_board, max_ceiling=6000, normal_budget=1600, min_floor=350)
    else:
        use_visits = int(current_ai_visits)
        tag = f"固定推演 ({use_visits}次)"

    last_dynamic_info = {"visits": use_visits, "tag": tag}

    # 2. 深度推演：AI 决策落子点
    if (opponent_engine == "katago" or current_board_size == 19) and katago_service.is_available():
        analysis = katago_service.analyze_board(game_board.move_history, board_size=game_board.size, komi=game_board.komi, max_visits=use_visits)
        if analysis and analysis.get("top_candidates"):
            action = analysis["top_candidates"][0]["move"]
        else:
            action = 'pass'
    else:
        # 自建 AlphaZero 引擎
        ai_agent.set_simulations(min(use_visits, 800))
        action = ai_agent.get_move(game_board, temp=0.0)

    # 3. AI 胜率追踪与认输判定
    current_eval = 50.0
    if (opponent_engine == "katago" or current_board_size == 19) and katago_service.is_available() and analysis:
        current_eval = analysis.get("white_winrate", 50.0) if game_board.current_player == WHITE else analysis.get("black_winrate", 50.0)
    ai_winrate_history.append(current_eval)

    # 三大认输策略判定
    if resign_mode == "smart":
        if len(ai_winrate_history) >= 5 and len(game_board.move_history) >= 28:
            recent_5 = ai_winrate_history[-5:]
            if all(w <= 1.0 for w in recent_5) and (max(recent_5) - min(recent_5) < 0.6):
                score = game_board.compute_score()
                if score > 6.0:
                    game_board.resign(game_board.current_player)
                    return get_state(recompute_eval=False)
    elif resign_mode == "quick":
        if current_eval < 2.0 and len(game_board.move_history) >= 20:
            game_board.resign(game_board.current_player)
            return get_state(recompute_eval=False)
    elif resign_mode == "never":
        pass

    # 4. AI 正式落子上盘
    if not game_board.is_valid_move(action):
        action = 'pass'
    game_board.play_move(action)

    # 5. 🎯 核心修复：AI 落子完成后，立即为【人类玩家的当前轮次】研判推荐选点与胜率！
    if not game_board.is_game_over():
        if referee_engine == "katago" and katago_service.is_available():
            # 为人类玩家生成前三选推荐着法与人类胜率
            human_analysis = katago_service.analyze_board(game_board.move_history, board_size=game_board.size, komi=game_board.komi, max_visits=min(use_visits, 1200))
            if human_analysis:
                human_analysis["dynamic_info"] = last_dynamic_info
                last_evaluation = human_analysis
        else:
            eval_az = ai_agent.get_evaluation(game_board)
            if eval_az:
                eval_az["dynamic_info"] = last_dynamic_info
                last_evaluation = eval_az

    return get_state(recompute_eval=False)

@app.post("/resign")
def resign():
    if not game_board.is_game_over():
        game_board.resign(player_color) # Resign current player's color
    return get_state(recompute_eval=False)

@app.post("/undo")
def undo():
    global game_board, ai_winrate_history, last_evaluation
    min_moves = current_handicap if current_handicap >= 2 else 0
    if len(game_board.move_history) <= min_moves:
        return get_state(recompute_eval=False)
        
    steps_to_undo = 2 if len(game_board.move_history) - min_moves >= 2 else 1
    new_history = game_board.move_history[:-steps_to_undo]
    ai_winrate_history = ai_winrate_history[:-steps_to_undo]
    
    komi = 0.5 if current_handicap >= 1 else current_komi
    new_board = GoBoard(size=game_board.size, komi=komi, handicap=current_handicap)
    actual_moves = new_history[min_moves:]
    for _, action in actual_moves:
        new_board.play_move(action)
        
    game_board = new_board
    last_evaluation = None
    return get_state(recompute_eval=True)

@app.post("/reset")
def reset_game():
    global game_board, ai_winrate_history, player_color, last_evaluation
    
    # Determine player color
    if color_mode == "random":
        player_color = random.choice([BLACK, WHITE])
    elif color_mode == "white":
        player_color = WHITE
    else:
        player_color = BLACK

    # Determine komi and handicap
    komi = current_komi
    if current_handicap == 1:
        komi = 0.5 # 让先
    elif current_handicap >= 2:
        komi = 0.5 # 让子

    game_board = GoBoard(size=current_board_size, komi=komi, handicap=current_handicap)
    ai_winrate_history = []
    last_evaluation = None
    
    # If it's AI's turn to play at game start (e.g. AI is Black when player is White, OR AI is White in handicap games), AI makes first move!
    if game_board.current_player != player_color:
        return ai_move()

    return get_state(recompute_eval=True)

from fastapi.staticfiles import StaticFiles
import os

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
