import os
import sys
import json
import subprocess
import threading
import time

class KataGoService:
    def __init__(self, max_visits=150):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.katago_dir = os.path.join(self.base_dir, 'katago')
        self.exe_path = os.path.join(self.katago_dir, 'katago.exe')
        self.model_path = os.path.join(self.katago_dir, 'kata9x9.bin.gz')
        self.cfg_path = os.path.join(self.katago_dir, 'analysis_9x9.cfg')
        self.max_visits = max_visits
        
        self.process = None
        self.lock = threading.Lock()
        self.query_counter = 0
        
        self._ensure_config()
        self._start_process()

    def _ensure_config(self):
        """Creates a streamlined 9x9 analysis config for KataGo."""
        if not os.path.exists(self.cfg_path):
            cfg_content = """# KataGo 9x9 Analysis Config
rules = chinese
komi = 6.5
maxVisits = 150
numSearchThreads = 4
nnCacheSizePowerOfTwo = 18
nnMutexPoolSizePowerOfTwo = 15
useOpenCL = true
openclDeviceToUse = 0
"""
            with open(self.cfg_path, 'w', encoding='utf-8') as f:
                f.write(cfg_content)

    def _start_process(self):
        if not os.path.exists(self.exe_path) or not os.path.exists(self.model_path):
            print(f"[KataGo] Warning: Binary or model missing. Exe: {self.exe_path}, Model: {self.model_path}")
            return

        cmd = [
            self.exe_path,
            "analysis",
            "-config", self.cfg_path,
            "-model", self.model_path
        ]
        
        try:
            print("[KataGo] Starting KataGo 9x9 Analysis Engine subprocess...")
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1
            )
            # Drain stderr in background to prevent pipe blocking
            def drain_stderr(proc):
                for line in proc.stderr:
                    pass
            threading.Thread(target=drain_stderr, args=(self.process,), daemon=True).start()
            print("[KataGo] KataGo engine successfully initialized and active!")
        except Exception as e:
            print(f"[KataGo] Failed to start process: {e}")
            self.process = None

    def is_available(self):
        return self.process is not None and self.process.poll() is None

    def analyze_board(self, move_history, board_size=9, komi=7.0, initial_stones=None, initial_player="B", max_visits=None):
        """
        Query KataGo Analysis Engine with move history and optional initial pre-placed stones.
        """
        if not self.is_available():
            return None

        # Convert move history to KataGo format [["B", "E5"], ["W", "D4"], ...]
        katago_moves = []
        for color, action in move_history:
            player_tag = "B" if color == 1 else "W"
            if action == 'pass':
                katago_moves.append([player_tag, "pass"])
            else:
                r, c = action
                # KataGo coordinate: Col A-J (skip I), Row 1-9 (1 is bottom, 9 is top)
                col_letters = "ABCDEFGHJKLMNOPQRST"
                col_char = col_letters[c]
                row_num = board_size - r
                katago_moves.append([player_tag, f"{col_char}{row_num}"])

        visits_to_use = max_visits if max_visits is not None else self.max_visits

        with self.lock:
            self.query_counter += 1
            query_id = f"q_{self.query_counter}"
            
            query = {
                "id": query_id,
                "rules": "chinese",
                "komi": komi,
                "boardXSize": board_size,
                "boardYSize": board_size,
                "moves": katago_moves,
                "includeOwnership": True,
                "maxVisits": visits_to_use
            }
            if initial_stones and len(initial_stones) > 0:
                query["initialStones"] = initial_stones
                query["initialPlayer"] = initial_player

            try:
                # Send query
                query_str = json.dumps(query) + "\n"
                self.process.stdin.write(query_str)
                self.process.stdin.flush()

                # Read response
                response_line = self.process.stdout.readline()
                if not response_line:
                    return None
                    
                data = json.loads(response_line)
                if data.get("id") != query_id:
                    return None

                root_info = data.get("rootInfo", {})
                move_infos = data.get("moveInfos", [])
                raw_own = data.get("ownership", [])

                # Parse KataGo metrics
                # In KataGo Analysis JSON Protocol, rootInfo.winrate is ALWAYS absolute Black winrate (0.0 - 1.0)
                # and rootInfo.scoreLead is ALWAYS absolute Black lead (>0: Black lead, <0: White lead)
                black_winrate = root_info.get("winrate", 0.5)
                white_winrate = 1.0 - black_winrate
                black_lead = root_info.get("scoreLead", 0.0)
                curr_player = root_info.get("currentPlayer", "B")

                # Parse candidate moves
                candidates = []
                col_letters = "ABCDEFGHJKLMNOPQRST"
                for m in move_infos:
                    move_str = m.get("move", "")
                    if move_str.lower() == "pass":
                        action = "pass"
                    elif len(move_str) >= 2:
                        c_char = move_str[0].upper()
                        r_num = int(move_str[1:])
                        c = col_letters.index(c_char)
                        r = board_size - r_num
                        action = (r, c)
                    else:
                        continue

                    raw_cand_wr = m.get("winrate", 0.5)
                    # Candidate display winrate for the player who is about to play
                    cand_player_winrate = raw_cand_wr if curr_player == "B" else (1.0 - raw_cand_wr)
                    m_lead = m.get("scoreLead", 0.0)
                    prior = m.get("prior", 0.0)
                    visits = m.get("visits", 0)

                    candidates.append({
                        "move": action,
                        "coord_str": move_str,
                        "winrate": round(float(cand_player_winrate) * 100, 1),
                        "black_winrate": round(float(raw_cand_wr) * 100, 1),
                        "score_lead": round(float(m_lead), 1),
                        "prob": round(float(prior) * 100, 1),
                        "visits": int(visits)
                    })

                # 2D Ownership Matrix & WildFox-style Discretized Territory Summary
                ownership_2d = []
                b_pts = 0
                w_pts = 0
                TERRITORY_THRESHOLD = 0.40
                
                if len(raw_own) == board_size * board_size:
                    for r in range(board_size):
                        row = []
                        for c in range(board_size):
                            val = float(raw_own[r * board_size + c])
                            row.append(round(val, 3))
                            if val >= TERRITORY_THRESHOLD:
                                b_pts += 1
                            elif val <= -TERRITORY_THRESHOLD:
                                w_pts += 1
                        ownership_2d.append(row)

                diff_pts = b_pts - (w_pts + komi)
                if abs(diff_pts) < 0.5:
                    fox_lead_txt = "双方势均力敌 (五五开)"
                elif diff_pts > 0:
                    fox_lead_txt = f"黑胜 {diff_pts:.1f} 目" + (" (细棋局面)" if diff_pts <= 2.5 else " (黑优势)")
                else:
                    fox_lead_txt = f"白胜 {abs(diff_pts):.1f} 目" + (" (细棋局面)" if abs(diff_pts) <= 2.5 else " (白优势)")

                # Standard Exact High-Precision Winrate & Score Lead (KataGo MCTS root)
                lead_exact = round(float(black_lead), 1)
                lead_exact_txt = f"黑领先 {lead_exact:.1f} 目" if lead_exact >= 0 else f"白领先 {abs(lead_exact):.1f} 目"

                return {
                    "black_winrate": round(float(black_winrate) * 100, 1),
                    "white_winrate": round(float(white_winrate) * 100, 1),
                    "value": round(float(black_winrate * 2.0 - 1.0), 3),
                    "score_lead": lead_exact,
                    "lead_text": lead_exact_txt,
                    "top_candidates": candidates[:5],
                    "ownership": ownership_2d,
                    "black_territory": b_pts,
                    "white_territory": w_pts,
                    "white_total": round(w_pts + komi, 1),
                    "black_total": round(b_pts, 1),
                    "fox_lead_text": fox_lead_txt,
                    "fox_lead": round(diff_pts, 1)
                }
            except Exception as e:
                print(f"[KataGo] Query error: {e}")
                return None

    def get_best_move(self, move_history, board_size=9, komi=7.0, max_visits=None):
        """Returns the best move calculated by KataGo."""
        analysis = self.analyze_board(move_history, board_size=board_size, komi=komi, max_visits=max_visits)
        if analysis and analysis.get("top_candidates"):
            return analysis["top_candidates"][0]["move"]
        return "pass"

    def close(self):
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None

# Global Singleton
katago_service = KataGoService(max_visits=150)
