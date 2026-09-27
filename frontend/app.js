const API_URL = (window.location.protocol.startsWith("http")) ? window.location.origin : "http://127.0.0.1:8000";
const CANVAS_SIZE = 480;
const MARGIN = 26;
let currentBoardSize = 9;

function getBoardSize() {
    if (gameState && gameState.board_size) return gameState.board_size;
    if (gameState && gameState.board && gameState.board.length) return gameState.board.length;
    return currentBoardSize || 9;
}

function getGridSize() {
    const size = getBoardSize();
    return (CANVAS_SIZE - 2 * MARGIN) / Math.max(1, size - 1);
}

const ALL_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'];

const canvas = document.getElementById('go-board');
const ctx = canvas.getContext('2d');

const chartCanvas = document.getElementById('winrate-chart');
const chartCtx = chartCanvas.getContext('2d');

// UI Elements
const blackWinrateEl = document.getElementById('black-winrate');
const whiteWinrateEl = document.getElementById('white-winrate');
const barBlackEl = document.getElementById('bar-black');
const barWhiteEl = document.getElementById('bar-white');
const situationBadgeEl = document.getElementById('situation-badge');
const scoreLeadTextEl = document.getElementById('score-lead-text');
const turnBadgeEl = document.getElementById('turn-badge');
const moveCountEl = document.getElementById('move-count');
const moveBadgeMiniEl = document.getElementById('move-badge-mini');
const blackCapturesEl = document.getElementById('black-captures');
const whiteCapturesEl = document.getElementById('white-captures');
const toggleHintsEl = document.getElementById('toggle-hints');
const toggleWinrateEl = document.getElementById('toggle-winrate');
const toggleTerritoryEl = document.getElementById('toggle-territory');
const winrateContainerEl = document.querySelector('.winrate-container');
const aiSpinnerEl = document.getElementById('ai-spinner');
const aiStatusTextEl = document.getElementById('ai-status-text');
const gameOverBannerEl = document.getElementById('game-over-banner');
const resultTitleEl = document.getElementById('result-title');
const resultDetailEl = document.getElementById('result-detail');

// AI 申请终局数子交互弹窗元素
const aiScoringModalEl = document.getElementById('ai-scoring-modal');
const modalLeadPillEl = document.getElementById('modal-lead-pill');
const btnScoringAgreeEl = document.getElementById('btn-scoring-agree');
const btnScoringDeclineEl = document.getElementById('btn-scoring-decline');
let userDeclinedScoringMove = -1; // 记录玩家上次拒绝数子的手数，避免在同一手重复弹窗

// Sidebar Elements
const candidateListEl = document.getElementById('candidate-list');
const aiCommentaryEl = document.getElementById('ai-commentary');
const netValueEl = document.getElementById('net-value');
const mctsSimsValEl = document.getElementById('mcts-sims-val');
const topEngineTagEl = document.getElementById('top-engine-tag');

let gameState = null;
let isAiThinking = false;
let hoveredIntersection = null; // {r, c}
let hoveredCandidatePoint = null; // [r, c] for candidate hover link
let winrateHistory = [50.0]; // Track Black win rate per move
let showTerritory = false;

// Coordinate format: (r, c) -> "E5"
function toCoordName(r, c) {
    const size = getBoardSize();
    const rowLabel = size - r;
    return `${ALL_COLS[c]}${rowLabel}`;
}

// ==========================================
// 🎵 Web Audio API 声音合成引擎
// ==========================================
let audioCtx = null;
function getAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

function playStoneSound() {
    try {
        const ctx = getAudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(420, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(140, ctx.currentTime + 0.08);
        gain.gain.setValueAtTime(0.7, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.08);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.08);
    } catch (e) {}
}

function playClockBeep(freq = 880, duration = 0.08, type = 'sine') {
    try {
        const ctx = getAudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        gain.gain.setValueAtTime(0.18, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + duration);
    } catch (e) {}
}

function playByoyomiWarning(secondsLeft) {
    if (secondsLeft <= 5 && secondsLeft > 0) {
        playClockBeep(secondsLeft === 1 ? 1200 : 960, 0.12, 'triangle');
    } else if (secondsLeft === 10) {
        playClockBeep(680, 0.15, 'sine');
    }
}

function playTimeoutDefeatSound() {
    try {
        [400, 320, 240].forEach((freq, i) => {
            setTimeout(() => playClockBeep(freq, 0.25, 'sawtooth'), i * 200);
        });
    } catch (e) {}
}

// ==========================================
// ⏱️ 对弈用时与读秒系统 (野狐 / 弈城 Byo-yomi)
// ==========================================
const TIME_PRESETS = {
    "rapid": { name: "快速对弈", mainTime: 60, byoyomiPeriods: 3, byoyomiTime: 20 },
    "blitz": { name: "闪电超快棋", mainTime: 0, byoyomiPeriods: 3, byoyomiTime: 10 },
    "standard": { name: "锦标标准", mainTime: 300, byoyomiPeriods: 3, byoyomiTime: 30 },
    "slow": { name: "慢棋深思", mainTime: 900, byoyomiPeriods: 3, byoyomiTime: 30 },
    "none": { name: "无限时", mainTime: 0, byoyomiPeriods: 0, byoyomiTime: 0 }
};

let currentTimePreset = "rapid";
let timerInterval = null;

let clockState = {
    black: { main: 60, periods: 3, byo: 20, inByo: false },
    white: { main: 60, periods: 3, byo: 20, inByo: false }
};

function formatSeconds(totalSecs) {
    const m = Math.floor(totalSecs / 60);
    const s = totalSecs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function initClock(presetKey) {
    if (presetKey) {
        currentTimePreset = presetKey;
    } else {
        const selectTimeControlEl = document.getElementById('select-time-control');
        if (selectTimeControlEl) currentTimePreset = selectTimeControlEl.value || "rapid";
    }

    const cfg = TIME_PRESETS[currentTimePreset] || TIME_PRESETS["rapid"];
    clockState = {
        black: {
            main: cfg.mainTime,
            periods: cfg.byoyomiPeriods,
            byo: cfg.byoyomiTime,
            inByo: cfg.mainTime === 0 && cfg.byoyomiPeriods > 0
        },
        white: {
            main: cfg.mainTime,
            periods: cfg.byoyomiPeriods,
            byo: cfg.byoyomiTime,
            inByo: cfg.mainTime === 0 && cfg.byoyomiPeriods > 0
        }
    };

    if (!timerInterval) {
        timerInterval = setInterval(tickClock, 1000);
    }
    updateClockUI();
}

function onMovePlacedClock(color) {
    const cfg = TIME_PRESETS[currentTimePreset] || TIME_PRESETS["rapid"];
    const side = color === 1 ? clockState.black : clockState.white;
    if (side.inByo) {
        side.byo = cfg.byoyomiTime;
    }
    updateClockUI();
}

function tickClock() {
    if (!gameState || gameState.is_game_over) return;
    if (currentTimePreset === "none") return;

    // 🛑 第一步未落子前（第0手），计时器保持就绪状态，不开始倒计时扣减
    const moveCount = (gameState.move_history && gameState.move_history.length) || 0;
    if (moveCount === 0) {
        updateClockUI();
        return;
    }

    const currColor = gameState.current_player; // 1: Black, -1: White
    const side = currColor === 1 ? clockState.black : clockState.white;
    const humanColor = (gameState && gameState.player_color) || 1;
    const isHuman = (currColor === humanColor);

    if (!side.inByo) {
        if (side.main > 0) {
            side.main--;
            if (side.main === 0) {
                if (side.periods > 0) {
                    side.inByo = true;
                    side.byo = TIME_PRESETS[currentTimePreset].byoyomiTime;
                    if (isHuman) playClockBeep(700, 0.25);
                } else {
                    handleTimeout(currColor);
                    return;
                }
            }
        }
    } else {
        if (side.byo > 0) {
            if (isHuman) playByoyomiWarning(side.byo);
            side.byo--;
            if (side.byo === 0) {
                side.periods--;
                if (side.periods <= 0) {
                    handleTimeout(currColor);
                    return;
                } else {
                    side.byo = TIME_PRESETS[currentTimePreset].byoyomiTime;
                    if (isHuman) playClockBeep(550, 0.2);
                }
            }
        }
    }
    updateClockUI();
}

function handleTimeout(loserColor) {
    if (!gameState || gameState.is_game_over) return;
    gameState.is_game_over = true;
    playTimeoutDefeatSound();

    const humanColor = (gameState && gameState.player_color) || 1;
    const winnerColor = -loserColor;
    gameState.score = loserColor === 1 ? -999.0 : 999.0;

    if (gameOverBannerEl) gameOverBannerEl.style.display = "flex";
    if (resultTitleEl) {
        resultTitleEl.innerText = loserColor === 1 ? "⏰ 黑方超时负！" : "⏰ 白方超时负！";
    }
    if (resultDetailEl) {
        resultDetailEl.innerText = loserColor === humanColor ? "很遗憾，您的读秒时间已耗尽，判为超时负。" : "AI 对弈引擎思考超时，玩家获得超时胜！";
    }
    if (aiStatusTextEl) {
        aiStatusTextEl.innerText = loserColor === humanColor ? "玩家超时负" : "AI 超时负";
    }
    updateUI();
}

function updateClockUI() {
    const blackTimeEl = document.getElementById('black-clock-time');
    const whiteTimeEl = document.getElementById('white-clock-time');
    const blackTagEl = document.getElementById('black-byoyomi-tag');
    const whiteTagEl = document.getElementById('white-byoyomi-tag');
    const blackBoxEl = document.getElementById('black-clock-box');
    const whiteBoxEl = document.getElementById('white-clock-box');

    if (!blackTimeEl || !whiteTimeEl) return;

    if (currentTimePreset === "none") {
        blackTimeEl.innerText = "不限时";
        whiteTimeEl.innerText = "不限时";
        if (blackTagEl) blackTagEl.style.display = "none";
        if (whiteTagEl) whiteTagEl.style.display = "none";
        if (blackBoxEl) blackBoxEl.classList.remove('active-clock-turn', 'in-byoyomi');
        if (whiteBoxEl) whiteBoxEl.classList.remove('active-clock-turn', 'in-byoyomi');
        return;
    } else {
        if (blackTagEl) blackTagEl.style.display = "inline-block";
        if (whiteTagEl) whiteTagEl.style.display = "inline-block";
    }

    const b = clockState.black;
    const w = clockState.white;

    if (!b.inByo) {
        blackTimeEl.innerText = formatSeconds(b.main);
        if (blackTagEl) blackTagEl.innerText = `${b.periods}次 ${TIME_PRESETS[currentTimePreset].byoyomiTime}s`;
        if (blackBoxEl) blackBoxEl.classList.remove('in-byoyomi');
    } else {
        blackTimeEl.innerText = `${b.byo}s`;
        if (blackTagEl) blackTagEl.innerText = `🔥 读秒(剩${b.periods}次)`;
        if (blackBoxEl) blackBoxEl.classList.add('in-byoyomi');
    }

    if (!w.inByo) {
        whiteTimeEl.innerText = formatSeconds(w.main);
        if (whiteTagEl) whiteTagEl.innerText = `${w.periods}次 ${TIME_PRESETS[currentTimePreset].byoyomiTime}s`;
        if (whiteBoxEl) whiteBoxEl.classList.remove('in-byoyomi');
    } else {
        whiteTimeEl.innerText = `${w.byo}s`;
        if (whiteTagEl) whiteTagEl.innerText = `🔥 读秒(剩${w.periods}次)`;
        if (whiteBoxEl) whiteBoxEl.classList.add('in-byoyomi');
    }

    if (gameState && !gameState.is_game_over) {
        const moveCount = (gameState.move_history && gameState.move_history.length) || 0;
        if (moveCount === 0) {
            // 第一步未落子：时钟静止就绪，不开启呼吸光圈
            if (blackBoxEl) blackBoxEl.classList.remove('active-clock-turn');
            if (whiteBoxEl) whiteBoxEl.classList.remove('active-clock-turn');
        } else if (gameState.current_player === 1) {
            if (blackBoxEl) blackBoxEl.classList.add('active-clock-turn');
            if (whiteBoxEl) whiteBoxEl.classList.remove('active-clock-turn');
        } else {
            if (whiteBoxEl) whiteBoxEl.classList.add('active-clock-turn');
            if (blackBoxEl) blackBoxEl.classList.remove('active-clock-turn');
        }
    } else {
        if (blackBoxEl) blackBoxEl.classList.remove('active-clock-turn');
        if (whiteBoxEl) whiteBoxEl.classList.remove('active-clock-turn');
    }
}

function gridToPixel(r, c) {
    const gridSize = getGridSize();
    return {
        x: MARGIN + c * gridSize,
        y: MARGIN + r * gridSize
    };
}

function pixelToGrid(x, y) {
    const size = getBoardSize();
    const gridSize = getGridSize();
    const c = Math.round((x - MARGIN) / gridSize);
    const r = Math.round((y - MARGIN) / gridSize);
    if (r >= 0 && r < size && c >= 0 && c < size) {
        const p = gridToPixel(r, c);
        const dist = Math.hypot(x - p.x, y - p.y);
        if (dist <= gridSize * 0.48) {
            return { r, c };
        }
    }
    return null;
}

// Fetch current state from backend
async function fetchState() {
    try {
        const response = await fetch(`${API_URL}/state`);
        gameState = await response.json();
        if (gameState.evaluation) {
            if (winrateHistory.length <= 1) {
                winrateHistory = [gameState.evaluation.black_winrate];
            }
        }
        const selectBoardSizeEl = document.getElementById('select-board-size');
        if (selectBoardSizeEl && gameState.board_size) {
            selectBoardSizeEl.value = gameState.board_size.toString();
        }
        const selectOpponentEngineEl = document.getElementById('select-opponent-engine');
        if (selectOpponentEngineEl) {
            const optAz = selectOpponentEngineEl.querySelector('option[value="alphazero"]');
            if (optAz) optAz.disabled = (gameState.board_size === 19);
            selectOpponentEngineEl.value = gameState.opponent_engine || "katago";
        }
        const selectRefereeEngineEl = document.getElementById('select-referee-engine');
        if (selectRefereeEngineEl) {
            const optRefAz = selectRefereeEngineEl.querySelector('option[value="alphazero"]');
            if (optRefAz) optRefAz.disabled = (gameState.board_size === 19);
            selectRefereeEngineEl.value = gameState.referee_engine || "katago";
        }
        const selectAiVisitsEl = document.getElementById('select-ai-visits');
        if (selectAiVisitsEl && gameState.ai_visits !== undefined) {
            selectAiVisitsEl.value = gameState.ai_visits.toString();
        }
        const selectKomiEl = document.getElementById('select-komi');
        if (selectKomiEl && gameState.komi !== undefined) {
            selectKomiEl.value = gameState.komi.toFixed(1);
        }
        const selectGameSpecEl = document.getElementById('select-game-spec');
        if (selectGameSpecEl && gameState.color_mode) {
            const h = gameState.handicap !== undefined ? gameState.handicap : 0;
            const specKey = `${gameState.color_mode}_${h}`;
            if (selectGameSpecEl.querySelector(`option[value="${specKey}"]`)) {
                selectGameSpecEl.value = specKey;
            } else {
                selectGameSpecEl.value = `${gameState.color_mode}_0`;
            }
        }
        const selectResignModeEl = document.getElementById('select-resign-mode');
        if (selectResignModeEl && gameState.resign_mode) {
            selectResignModeEl.value = gameState.resign_mode;
        }
        updateUI();
    } catch (error) {
        if (aiStatusTextEl) {
            aiStatusTextEl.innerText = "无法连接至后端服务 (请先在终端运行 uvicorn backend.main:app)";
        }
        console.error(error);
    }
}

// Draw the Go board and elements
function drawBoard() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const size = getBoardSize();
    const gridSize = getGridSize();
    
    // Draw wood background with delicate texture gradient
    const bgGrad = ctx.createRadialGradient(
        canvas.width * 0.45, canvas.height * 0.45, 20,
        canvas.width / 2, canvas.height / 2, canvas.width * 0.7
    );
    bgGrad.addColorStop(0, "#e8c37d");
    bgGrad.addColorStop(1, "#cf9e4e");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw coordinate labels
    ctx.fillStyle = "#5c3c10";
    ctx.font = size === 19 ? "bold 8px Inter, sans-serif" : (size === 13 ? "bold 9.5px Inter, sans-serif" : "bold 11px Inter, sans-serif");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < size; i++) {
        const p = gridToPixel(i, i);
        ctx.fillText(ALL_COLS[i], p.x, MARGIN / 2);
        ctx.fillText(ALL_COLS[i], p.x, CANVAS_SIZE - MARGIN / 2);
        const rowLabel = (size - i).toString();
        ctx.fillText(rowLabel, MARGIN / 2, p.y);
        ctx.fillText(rowLabel, CANVAS_SIZE - MARGIN / 2, p.y);
    }

    // Draw board grid lines
    ctx.strokeStyle = "#38240a";
    ctx.lineWidth = size === 19 ? 1.0 : (size === 13 ? 1.2 : 1.6);

    for (let i = 0; i < size; i++) {
        const startH = gridToPixel(i, 0);
        const endH = gridToPixel(i, size - 1);
        ctx.beginPath();
        ctx.moveTo(startH.x, startH.y);
        ctx.lineTo(endH.x, endH.y);
        ctx.stroke();

        const startV = gridToPixel(0, i);
        const endV = gridToPixel(size - 1, i);
        ctx.beginPath();
        ctx.moveTo(startV.x, startV.y);
        ctx.lineTo(endV.x, endV.y);
        ctx.stroke();
    }

    // Draw star points (4 corners + Tengen + 4 sides)
    let starPoints = [];
    if (size === 19) {
        starPoints = [
            [3, 3], [3, 9], [3, 15],
            [9, 3], [9, 9], [9, 15],
            [15, 3], [15, 9], [15, 15]
        ];
    } else if (size === 13) {
        starPoints = [
            [3, 3], [3, 9], [9, 3], [9, 9], [6, 6],
            [3, 6], [9, 6], [6, 3], [6, 9]
        ];
    } else {
        starPoints = [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]];
    }

    ctx.fillStyle = "#38240a";
    const starR = size === 19 ? 2.5 : (size === 13 ? 3.0 : 3.8);
    for (let [r, c] of starPoints) {
        const p = gridToPixel(r, c);
        ctx.beginPath();
        ctx.arc(p.x, p.y, starR, 0, 2 * Math.PI);
        ctx.fill();
    }

    if (!gameState || !gameState.board) return;

    // Draw Territory Ownership Overlay if enabled
    if (showTerritory && gameState && gameState.territory && gameState.territory.ownership) {
        drawTerritoryOverlay(gameState.territory.ownership);
    }

    // Draw placed stones
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            if (gameState.board[r] && gameState.board[r][c] !== 0) {
                const stone = gameState.board[r][c];
                drawStone(r, c, stone);
            }
        }
    }

    // Draw Dead Stones Overlay if territory view is active
    if (showTerritory && gameState && gameState.territory && gameState.territory.ownership) {
        drawDeadStonesOverlay(gameState.territory.ownership);
    }

    // Draw AI Candidate Badges displaying Expected Win Rate if toggle is checked and it is human turn
    const humanColor = (gameState && gameState.player_color) || 1;
    const isHumanTurn = (gameState.current_player === humanColor) && !isAiThinking;
    if (toggleHintsEl.checked && isHumanTurn && gameState.evaluation && gameState.evaluation.top_candidates && !gameState.is_game_over && !showTerritory) {
        drawAiHints(gameState.evaluation.top_candidates);
    }

    // Draw hovered candidate highlight linked from right sidebar
    if (hoveredCandidatePoint && !gameState.is_game_over) {
        drawCandidateHoverTarget(hoveredCandidatePoint[0], hoveredCandidatePoint[1]);
    }

    // Draw ghost stone on hover (when it's player's turn and intersection is empty)
    if (hoveredIntersection && !isAiThinking && !gameState.is_game_over && gameState.current_player === humanColor) {
        const { r, c } = hoveredIntersection;
        if (gameState.board[r] && gameState.board[r][c] === 0) {
            drawGhostStone(r, c, humanColor);
        }
    }

    // Highlight last move with contrasting pulsating marker
    if (gameState.last_move && gameState.last_move !== 'pass') {
        const [r, c] = gameState.last_move;
        const p = gridToPixel(r, c);
        const stoneColor = (gameState.board[r] && gameState.board[r][c]) || 1;

        ctx.save();
        ctx.strokeStyle = stoneColor === 1 ? "#ff6b6b" : "#4dabf7";
        ctx.lineWidth = size === 13 ? 2.2 : 2.8;
        ctx.beginPath();
        ctx.arc(p.x, p.y, gridSize * 0.22, 0, 2 * Math.PI);
        ctx.stroke();

        // Outer subtle glow ring
        ctx.strokeStyle = stoneColor === 1 ? "rgba(255, 107, 107, 0.4)" : "rgba(77, 171, 247, 0.4)";
        ctx.lineWidth = size === 13 ? 3.5 : 4.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, gridSize * 0.28, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
    }
}

function drawStone(r, c, color) {
    const p = gridToPixel(r, c);
    const gridSize = getGridSize();
    const radius = gridSize * 0.46;

    // Drop Shadow
    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x + 2, p.y + 2.5, radius, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fill();
    ctx.restore();

    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);

    if (color === 1) { // Black stone with 3D gradient
        const grad = ctx.createRadialGradient(p.x - radius/3, p.y - radius/3, radius/10, p.x, p.y, radius);
        grad.addColorStop(0, '#555');
        grad.addColorStop(0.3, '#262a33');
        grad.addColorStop(0.8, '#111317');
        grad.addColorStop(1, '#050608');
        ctx.fillStyle = grad;
    } else { // White stone with pearl luster
        const grad = ctx.createRadialGradient(p.x - radius/3, p.y - radius/3, radius/10, p.x, p.y, radius);
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.65, '#eaedf2');
        grad.addColorStop(0.9, '#cdd3df');
        grad.addColorStop(1, '#b8bfcf');
        ctx.fillStyle = grad;
        ctx.strokeStyle = '#a8b0c2';
        ctx.lineWidth = 0.8;
        ctx.stroke();
    }
    ctx.fill();
}

function drawGhostStone(r, c, color) {
    const p = gridToPixel(r, c);
    const gridSize = getGridSize();
    const radius = gridSize * 0.46;

    ctx.save();
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color === 1 ? '#000000' : '#ffffff';
    ctx.fill();

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = color === 1 ? '#fff' : '#000';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
}

// Draw candidate hover ring linked from sidebar
function drawCandidateHoverTarget(r, c) {
    const p = gridToPixel(r, c);
    const gridSize = getGridSize();
    ctx.save();
    
    // Outer animated target ring
    ctx.beginPath();
    ctx.arc(p.x, p.y, gridSize * 0.44, 0, 2 * Math.PI);
    ctx.strokeStyle = "#88c0d0";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(p.x, p.y, gridSize * 0.52, 0, 2 * Math.PI);
    ctx.strokeStyle = "rgba(136, 192, 208, 0.4)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
}

// Draw Territory Ownership Overlay (野狐围棋风格：清晰实地小方块标目，未定地不画杂色)
function drawTerritoryOverlay(grid) {
    if (!grid) return;
    const size = getBoardSize();
    const gridSize = getGridSize();
    const sqSize = Math.max(3.5, gridSize * 0.32); // 经典野狐目数实心方块尺寸
    const half = sqSize / 2;

    ctx.save();
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            if (!grid[r]) continue;
            const val = grid[r][c];
            
            // 仅对未落子的纯实地点绘制野狐式地盘小方块
            const hasStone = gameState && gameState.board && gameState.board[r] && gameState.board[r][c] !== 0;
            if (hasStone) continue;

            const p = gridToPixel(r, c);

            if (val >= 0.38) { // 确定黑棋实地 (1目)
                ctx.fillStyle = "#1e222b";
                ctx.strokeStyle = "#4c566a";
                ctx.lineWidth = 1.2;
                ctx.fillRect(p.x - half, p.y - half, sqSize, sqSize);
                ctx.strokeRect(p.x - half, p.y - half, sqSize, sqSize);
            } else if (val <= -0.38) { // 确定白棋实地 (1目)
                ctx.fillStyle = "#ffffff";
                ctx.strokeStyle = "#4c566a";
                ctx.lineWidth = 1.2;
                ctx.fillRect(p.x - half, p.y - half, sqSize, sqSize);
                ctx.strokeRect(p.x - half, p.y - half, sqSize, sqSize);
            }
            // 未定地 / 公共中腹（-0.38 < val < 0.38）：完全保持棋盘原貌，干净整洁
        }
    }
    ctx.restore();
}

// Draw Dead Stones Marker (野狐风格醒目死子红叉标记 ❌)
function drawDeadStonesOverlay(grid) {
    if (!grid || !gameState) return;
    const size = getBoardSize();
    const gridSize = getGridSize();
    ctx.save();
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            if (!gameState.board[r] || gameState.board[r][c] === 0) continue;
            const stone = gameState.board[r][c];
            if (!grid[r]) continue;
            const val = grid[r][c];
            
            // 黑子在白领地(val <= -0.30) 或 白子在黑领地(val >= 0.30) 判定为死子
            const isDead = (stone === 1 && val <= -0.30) || (stone === -1 && val >= 0.30);
            if (isDead) {
                const p = gridToPixel(r, c);
                const s = Math.max(3.0, gridSize * 0.22);

                // 阴影红叉
                ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
                ctx.lineWidth = size === 19 ? 3.0 : 4.0;
                ctx.beginPath();
                ctx.moveTo(p.x - s, p.y - s);
                ctx.lineTo(p.x + s, p.y + s);
                ctx.moveTo(p.x + s, p.y - s);
                ctx.lineTo(p.x - s, p.y + s);
                ctx.stroke();

                // 亮红主叉
                ctx.strokeStyle = "#ff3b30";
                ctx.lineWidth = size === 19 ? 1.8 : 2.5;
                ctx.beginPath();
                ctx.moveTo(p.x - s, p.y - s);
                ctx.lineTo(p.x + s, p.y + s);
                ctx.moveTo(p.x + s, p.y - s);
                ctx.lineTo(p.x - s, p.y + s);
                ctx.stroke();
            }
        }
    }
    ctx.restore();
}

// Draw AI Candidate Badges on Board showing EXPECTED WIN RATE (%)
function drawAiHints(candidates) {
    const colors = [
        { bg: "rgba(94, 129, 172, 0.92)", border: "#88c0d0", text: "#fff" }, // Rank 1: Cyan Blue
        { bg: "rgba(163, 190, 140, 0.92)", border: "#a3be8c", text: "#111" }, // Rank 2: Green
        { bg: "rgba(235, 203, 139, 0.92)", border: "#ebcb8b", text: "#111" }  // Rank 3: Gold
    ];
    const size = getBoardSize();
    const gridSize = getGridSize();

    candidates.forEach((cand, idx) => {
        if (!cand.move || cand.move === 'pass') return;
        const [r, c] = cand.move;
        if (!gameState.board[r] || gameState.board[r][c] !== 0) return; // Skip occupied
        if (idx >= colors.length) return;

        const p = gridToPixel(r, c);
        const color = colors[idx];
        const radius = gridSize * 0.38;

        ctx.save();
        // Glowing halo
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color.bg;
        ctx.fill();
        ctx.strokeStyle = color.border;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw Expected Win Rate % (KataGo style)
        ctx.fillStyle = color.text;
        ctx.font = size === 19 ? "bold 7px 'JetBrains Mono', sans-serif" : (size === 13 ? "bold 8.5px 'JetBrains Mono', sans-serif" : "bold 10px 'JetBrains Mono', sans-serif");
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const displayVal = cand.winrate !== undefined ? `${Math.round(cand.winrate)}%` : `${Math.round(cand.prob)}%`;
        ctx.fillText(displayVal, p.x, p.y);
        ctx.restore();
    });
}

// Draw Win Rate History Chart with Grid & Bezier smoothing
function drawWinrateChart() {
    const w = chartCanvas.width;
    const h = chartCanvas.height;
    chartCtx.clearRect(0, 0, w, h);

    if (winrateHistory.length === 0) return;

    // Background
    chartCtx.fillStyle = "rgba(0, 0, 0, 0.25)";
    chartCtx.fillRect(0, 0, w, h);

    // Grid lines at 75%, 25%
    chartCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    chartCtx.lineWidth = 1;
    [0.25, 0.75].forEach(ratio => {
        const y = h * (1 - ratio);
        chartCtx.beginPath();
        chartCtx.moveTo(0, y);
        chartCtx.lineTo(w, y);
        chartCtx.stroke();
    });

    // 50% Center Line (Fairness baseline)
    const midY = h / 2;
    chartCtx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    chartCtx.setLineDash([4, 4]);
    chartCtx.beginPath();
    chartCtx.moveTo(0, midY);
    chartCtx.lineTo(w, midY);
    chartCtx.stroke();
    chartCtx.setLineDash([]);

    if (!winrateHistory || winrateHistory.length === 0) return;

    // Plot points
    const points = [];
    const stepX = winrateHistory.length > 1 ? w / (winrateHistory.length - 1) : w;

    winrateHistory.forEach((winRate, idx) => {
        const x = idx * stepX;
        const clamped = Math.max(0, Math.min(100, winRate));
        const y = h - 6 - ((clamped / 100) * (h - 12));
        points.push({ x, y });
    });

    if (points.length === 1) {
        chartCtx.beginPath();
        chartCtx.arc(points[0].x / 2, points[0].y, 3.5, 0, 2 * Math.PI);
        chartCtx.fillStyle = "#88c0d0";
        chartCtx.fill();
        return;
    }

    // Fill area under curve with smooth bezier
    chartCtx.beginPath();
    chartCtx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        const p0 = points[i - 1];
        const p1 = points[i];
        const cx = (p0.x + p1.x) / 2;
        chartCtx.bezierCurveTo(cx, p0.y, cx, p1.y, p1.x, p1.y);
    }
    chartCtx.lineTo(points[points.length - 1].x, midY);
    chartCtx.lineTo(points[0].x, midY);
    chartCtx.closePath();

    const fillGrad = chartCtx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, "rgba(136, 192, 208, 0.35)");
    fillGrad.addColorStop(0.5, "rgba(136, 192, 208, 0.05)");
    fillGrad.addColorStop(1, "rgba(235, 203, 139, 0.35)");
    chartCtx.fillStyle = fillGrad;
    chartCtx.fill();

    // Line Path with bezier smoothing
    chartCtx.beginPath();
    chartCtx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        const p0 = points[i - 1];
        const p1 = points[i];
        const cx = (p0.x + p1.x) / 2;
        chartCtx.bezierCurveTo(cx, p0.y, cx, p1.y, p1.x, p1.y);
    }
    chartCtx.strokeStyle = "#88c0d0";
    chartCtx.lineWidth = 2.2;
    chartCtx.stroke();

    // Last point dot with pulsing glow
    const lastP = points[points.length - 1];
    chartCtx.beginPath();
    chartCtx.arc(lastP.x, lastP.y, 4, 0, 2 * Math.PI);
    chartCtx.fillStyle = "#88c0d0";
    chartCtx.fill();
    chartCtx.strokeStyle = "#fff";
    chartCtx.lineWidth = 1.5;
    chartCtx.stroke();
}

// Update Sidebar Candidate List with both Win Rate and Policy Recommendation %
function updateSidebarCandidates(candidates) {
    if (!candidates || candidates.length === 0) {
        candidateListEl.innerHTML = `
            <div class="candidate-placeholder">
                <span class="placeholder-icon">⏳</span>
                <span>暂无选点数据</span>
            </div>
        `;
        return;
    }

    let html = "";
    candidates.forEach((cand, idx) => {
        if (!cand.move) return;
        const coord = cand.move === 'pass' ? 'Pass' : toCoordName(cand.move[0], cand.move[1]);
        const rankClass = idx === 0 ? 'rank-1' : idx === 1 ? 'rank-2' : idx === 2 ? 'rank-3' : 'rank-other';
        const winrateVal = cand.winrate !== undefined ? cand.winrate : 50.0;
        const probVal = cand.prob !== undefined ? cand.prob : 0.0;
        const visitsVal = cand.visits ? `${cand.visits}v` : '';
        const moveDataAttr = (cand.move && cand.move !== 'pass') ? `${cand.move[0]},${cand.move[1]}` : '';

        html += `
            <div class="candidate-item" data-move="${moveDataAttr}">
                <span class="rank-badge ${rankClass}">#${idx + 1}</span>
                <span class="cand-coord">${coord}</span>
                <div class="cand-stats">
                    <div class="cand-stats-top">
                        <span class="cand-winrate-txt">${winrateVal}% 胜率</span>
                        <span class="cand-prob-txt">${probVal}% 推荐 ${visitsVal}</span>
                    </div>
                    <div class="cand-bar-bg">
                        <div class="cand-bar-fill" style="width: ${winrateVal}%;"></div>
                    </div>
                </div>
            </div>
        `;
    });

    candidateListEl.innerHTML = html;

    // Add hover listener for candidate items to highlight on board
    const items = candidateListEl.querySelectorAll('.candidate-item');
    items.forEach(item => {
        item.addEventListener('mouseenter', () => {
            const moveAttr = item.getAttribute('data-move');
            if (moveAttr) {
                const [r, c] = moveAttr.split(',').map(Number);
                hoveredCandidatePoint = [r, c];
                drawBoard();
            }
        });
        item.addEventListener('mouseleave', () => {
            hoveredCandidatePoint = null;
            drawBoard();
        });
    });
}

function updateUI() {
    if (!gameState) return;

    // 同步控制顶部 AI 胜率分析条显示隐藏
    if (toggleWinrateEl && winrateContainerEl) {
        winrateContainerEl.style.display = toggleWinrateEl.checked ? 'flex' : 'none';
    }

    const humanColor = (gameState && gameState.player_color) || 1;

    drawBoard();

    // Dynamic Player Tag labels (Black vs White role)
    const blackTagEl = document.getElementById('black-player-tag');
    const whiteTagEl = document.getElementById('white-player-tag');
    if (blackTagEl) blackTagEl.innerText = humanColor === 1 ? "黑方 (玩家)" : "黑方 (AI)";
    if (whiteTagEl) whiteTagEl.innerText = humanColor === -1 ? "白方 (玩家)" : "白方 (AI)";

    // Score Lead Text (目数差更新与色彩联动)
    if (scoreLeadTextEl) {
        let leadVal = null;
        if (gameState && gameState.evaluation && gameState.evaluation.score_lead !== undefined && gameState.evaluation.score_lead !== null) {
            leadVal = Number(gameState.evaluation.score_lead);
        } else if (gameState && gameState.territory && gameState.territory.lead !== undefined && gameState.territory.lead !== null) {
            leadVal = Number(gameState.territory.lead);
        }

        if (leadVal !== null && !isNaN(leadVal)) {
            if (Math.abs(leadVal) < 0.2) {
                scoreLeadTextEl.innerText = "盘面: 势均力敌 (0.0目)";
                scoreLeadTextEl.style.color = "var(--nord-cyan)";
            } else if (leadVal > 0) {
                scoreLeadTextEl.innerText = `黑领先 ${leadVal.toFixed(1)} 目`;
                scoreLeadTextEl.style.color = "#88c0d0";
            } else {
                scoreLeadTextEl.innerText = `白领先 ${Math.abs(leadVal).toFixed(1)} 目`;
                scoreLeadTextEl.style.color = "#ebcb8b";
            }
        } else {
            scoreLeadTextEl.innerText = "盘面: 势均力敌 (0.0目)";
            scoreLeadTextEl.style.color = "var(--nord-cyan)";
        }
    }

    // Win Rate & Evaluation update
    if (gameState.evaluation) {
        const bWin = gameState.evaluation.black_winrate;
        const wWin = gameState.evaluation.white_winrate;
        blackWinrateEl.innerText = `${bWin.toFixed(1)}%`;
        whiteWinrateEl.innerText = `${wWin.toFixed(1)}%`;
        barBlackEl.style.width = `${bWin}%`;
        barWhiteEl.style.width = `${wWin}%`;

        // Situation Judgment Badge & Commentary
        let commentary = "";
        if (bWin >= 90) {
            situationBadgeEl.innerText = humanColor === 1 ? "黑棋胜势 (玩家绝对优势)" : "黑棋胜势 (AI绝对优势)";
            situationBadgeEl.style.color = "#81a1c1";
            commentary = humanColor === 1 ? "您执黑占据绝对优势，大龙安定且实地遥遥领先，已锁定胜局！" : "AI执黑占据绝对优势，大龙安定且实地遥遥领先，翻盘难度极大。";
        } else if (bWin >= 75) {
            situationBadgeEl.innerText = humanColor === 1 ? "黑棋大优 (玩家领先)" : "黑棋大优 (AI领先)";
            situationBadgeEl.style.color = "#88c0d0";
            commentary = humanColor === 1 ? "您局面主动，厚势与实地兼备，保持稳健即可获胜。" : "AI执黑局面主动，厚势与实地兼备，请寻找局部逆转机会。";
        } else if (bWin >= 55) {
            situationBadgeEl.innerText = humanColor === 1 ? "黑棋稍优 (玩家微弱优势)" : "黑棋稍优 (AI微弱优势)";
            situationBadgeEl.style.color = "#a3be8c";
            commentary = "黑棋略占上风，局势处于关键争夺期，注意要点与官子。";
        } else if (bWin >= 45) {
            situationBadgeEl.innerText = "势均力敌 (五五开)";
            situationBadgeEl.style.color = "#eceff4";
            commentary = "双方势均力敌，局势极为胶着，接下来的每一步行棋都至关重要。";
        } else if (bWin >= 25) {
            situationBadgeEl.innerText = humanColor === -1 ? "白棋稍优 (玩家微弱优势)" : "白棋稍优 (AI微弱优势)";
            situationBadgeEl.style.color = "#ebcb8b";
            commentary = "白棋略占优势，黑棋需要寻找破局机会或发起反击。";
        } else if (bWin >= 10) {
            situationBadgeEl.innerText = humanColor === -1 ? "白棋大优 (玩家领先)" : "白棋大优 (AI领先)";
            situationBadgeEl.style.color = "#d08770";
            commentary = humanColor === -1 ? "您执白确立明显优势，注意防范黑棋拼死反扑。" : "AI执白确立明显优势，黑棋需在局部寻求复杂战斗制造乱局。";
        } else {
            situationBadgeEl.innerText = humanColor === -1 ? "白棋胜势 (玩家绝对优势)" : "白棋胜势 (AI绝对优势)";
            situationBadgeEl.style.color = "#bf616a";
            commentary = humanColor === -1 ? "您执白牢牢掌控胜势，各处死活均已成定局，胜利在望！" : "AI执白牢牢掌控盘面胜势，各处死活均已成定局。";
        }

        aiCommentaryEl.innerText = commentary;
        if (netValueEl) {
            netValueEl.innerText = gameState.evaluation.value !== undefined ? gameState.evaluation.value.toFixed(3) : "0.000";
        }

        // Update Sidebar Candidates only when Hints toggle is enabled and it is human turn
        const candidateSection = document.querySelector('.candidate-section');
        if (candidateSection) {
            candidateSection.style.display = toggleHintsEl.checked ? 'flex' : 'none';
        }
        const isHumanTurn = (gameState.current_player === humanColor) && !isAiThinking;
        if (toggleHintsEl.checked) {
            if (isHumanTurn && gameState.evaluation && gameState.evaluation.top_candidates) {
                updateSidebarCandidates(gameState.evaluation.top_candidates);
            } else {
                updateSidebarCandidates([]);
            }
        }
    }

    // Update Win Rate History Chart
    drawWinrateChart();

    // Move count & captures
    const mCount = gameState.move_count || 0;
    moveCountEl.innerText = `${mCount} 手`;
    if (moveBadgeMiniEl) moveBadgeMiniEl.innerText = `${mCount} 手`;

    if (gameState.captures) {
        blackCapturesEl.innerText = gameState.captures.black;
        whiteCapturesEl.innerText = gameState.captures.white;
    }

    // Turn indicator & Game Over banner
    if (gameState.is_game_over) {
        turnBadgeEl.className = "turn-indicator";
        turnBadgeEl.innerHTML = `<span class="turn-text">对局结束</span>`;
        gameOverBannerEl.style.display = "flex";

        // Auto show territory counting card upon game over
        if (territoryCardEl && gameState.territory) {
            territoryCardEl.style.display = 'block';
            showTerritory = true;
            if (toggleTerritoryEl) toggleTerritoryEl.checked = true;
        }

        const score = gameState.score;
        const komiText = gameState.komi !== undefined ? gameState.komi : 7.0;
        if (score === 999.0) {
            resultTitleEl.innerText = "🏆 黑方中盘胜！";
            resultDetailEl.innerText = humanColor === 1 ? "白方 (AI) 认输，玩家执黑获得中盘胜利。" : "白方 (玩家) 认输，AI 执黑获得中盘胜利。";
        } else if (score === -999.0) {
            resultTitleEl.innerText = "白方中盘胜！";
            resultDetailEl.innerText = humanColor === -1 ? "黑方 (AI) 认输，玩家执白获得中盘胜利。" : "黑方 (玩家) 认输，AI 执白获得中盘胜利。";
        } else if (score !== null && score !== undefined) {
            const absScore = Math.abs(score).toFixed(1);
            if (score > 0) {
                resultTitleEl.innerText = `🏆 黑方胜 ${absScore} 目！`;
                resultDetailEl.innerText = `终局裁判数子：黑棋 ${gameState.territory ? gameState.territory.black_territory : ''} 目，白棋 ${gameState.territory ? gameState.territory.white_total : ''} 目 (含贴目 ${komiText} 目)，黑方胜 ${absScore} 目。`;
            } else if (score < 0) {
                resultTitleEl.innerText = `白方胜 ${absScore} 目！`;
                resultDetailEl.innerText = `终局裁判数子：黑棋 ${gameState.territory ? gameState.territory.black_territory : ''} 目，白棋 ${gameState.territory ? gameState.territory.white_total : ''} 目 (含贴目 ${komiText} 目)，白方胜 ${absScore} 目。`;
            } else {
                resultTitleEl.innerText = "平局";
                resultDetailEl.innerText = "双方和棋";
            }
        }
        aiStatusTextEl.innerText = "对局已结束 (数子判胜完成)";
    } else {
        gameOverBannerEl.style.display = "none";
        const isHuman = (gameState.current_player === humanColor);
        if (gameState.current_player === 1) {
            turnBadgeEl.className = "turn-indicator black-turn";
            turnBadgeEl.innerHTML = `
                <span class="stone-bullet black-bullet"></span>
                <span class="turn-text">${isHuman ? "黑方落子 (玩家)" : "黑方落子 (AI)"}</span>
            `;
        } else {
            turnBadgeEl.className = "turn-indicator white-turn";
            turnBadgeEl.innerHTML = `
                <span class="stone-bullet white-bullet"></span>
                <span class="turn-text">${isHuman ? "白方落子 (玩家)" : "白方落子 (AI)"}</span>
            `;
        }
        if (isHuman) {
            aiStatusTextEl.innerText = "等待玩家落子...";
        } else {
            aiStatusTextEl.innerText = "等待 AI 落子...";
        }
    }

    // AI 提议终局数子弹窗控制
    if (aiScoringModalEl) {
        if (gameState.is_game_over) {
            aiScoringModalEl.style.display = 'none';
        } else if (gameState.ai_suggests_scoring && userDeclinedScoringMove !== gameState.move_count) {
            aiScoringModalEl.style.display = 'flex';
            if (modalLeadPillEl) {
                let leadTxt = "双方势均力敌";
                if (gameState.evaluation && gameState.evaluation.lead_text) {
                    leadTxt = gameState.evaluation.lead_text;
                } else if (gameState.territory && gameState.territory.lead_text) {
                    leadTxt = gameState.territory.lead_text;
                }
                modalLeadPillEl.innerText = `📊 当前研判：${leadTxt}`;
            }
        } else {
            aiScoringModalEl.style.display = 'none';
        }
    }

    // Update Territory Card numbers dynamically (野狐围棋风格格式化)
    if (gameState.territory) {
        const terrBlackEl = document.getElementById('terr-black');
        const terrWhiteEl = document.getElementById('terr-white');
        const terrLeadEl = document.getElementById('terr-lead');
        
        const bPts = parseInt(gameState.territory.black_territory) || 0;
        const wPts = parseInt(gameState.territory.white_territory) || 0;
        const komi = parseFloat(gameState.territory.komi) || 0.0;
        const wTotal = (wPts + komi).toFixed(1);

        if (terrBlackEl) terrBlackEl.innerText = `${bPts} 目`;
        if (terrWhiteEl) {
            if (komi > 0) {
                terrWhiteEl.innerText = `${wPts} 目 (含贴目 ${komi.toFixed(1)} = ${wTotal} 目)`;
            } else {
                terrWhiteEl.innerText = `${wPts} 目`;
            }
        }
        if (terrLeadEl) {
            terrLeadEl.innerText = gameState.territory.lead_text;
            terrLeadEl.style.color = gameState.territory.lead >= 0 ? "#88c0d0" : "#ebcb8b";
        }
    }

    // Button states
    const disableControls = isAiThinking || gameState.is_game_over;
    const btnReqScoring = document.getElementById('btn-request-scoring');
    if (btnReqScoring) btnReqScoring.disabled = disableControls;
    document.getElementById('btn-pass').disabled = disableControls;
    document.getElementById('btn-undo').disabled = isAiThinking || (gameState.move_count === 0);
    document.getElementById('btn-resign').disabled = isAiThinking || gameState.is_game_over;
}

// Mouse movement for hover ghost stone
canvas.addEventListener('mousemove', (e) => {
    const humanColor = (gameState && gameState.player_color) || 1;
    if (isAiThinking || !gameState || gameState.is_game_over || gameState.current_player !== humanColor) {
        hoveredIntersection = null;
        drawBoard();
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    hoveredIntersection = pixelToGrid(x, y);
    drawBoard();
});

canvas.addEventListener('mouseleave', () => {
    hoveredIntersection = null;
    drawBoard();
});

// Canvas click to make a move
canvas.addEventListener('click', (e) => {
    const humanColor = (gameState && gameState.player_color) || 1;
    if (isAiThinking || !gameState || gameState.is_game_over || gameState.current_player !== humanColor) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const gridPos = pixelToGrid(x, y);

    if (gridPos) {
        makeMove(gridPos.r, gridPos.c);
    }
});

// Hint checkbox toggle
toggleHintsEl.addEventListener('change', () => {
    drawBoard();
    const candidateSection = document.querySelector('.candidate-section');
    if (candidateSection) {
        candidateSection.style.display = toggleHintsEl.checked ? 'flex' : 'none';
    }
});

// Winrate Bar toggle (顶部 AI 胜率分析条开关)
if (toggleWinrateEl && winrateContainerEl) {
    toggleWinrateEl.addEventListener('change', () => {
        winrateContainerEl.style.display = toggleWinrateEl.checked ? 'flex' : 'none';
    });
}

// 实时形势判断热力图开关
if (toggleTerritoryEl) {
    toggleTerritoryEl.addEventListener('change', () => {
        showTerritory = toggleTerritoryEl.checked;
        if (territoryCardEl) {
            territoryCardEl.style.display = showTerritory ? 'block' : 'none';
        }
        if (showTerritory) {
            updateUI();
        } else {
            drawBoard();
        }
    });
}

async function triggerAI() {
    if (!gameState || gameState.is_game_over) return;

    isAiThinking = true;
    aiSpinnerEl.style.display = "flex";
    aiStatusTextEl.innerText = "AI 思考中 (KataGo / ResNet 深度搜索)...";
    updateUI();

    try {
        const response = await fetch(`${API_URL}/ai_move`, { method: 'POST' });
        gameState = await response.json();
        playStoneSound();
        if (gameState.evaluation) {
            winrateHistory.push(gameState.evaluation.black_winrate);
        }
        if (gameState.evaluation && gameState.evaluation.dynamic_info && aiStatusTextEl) {
            const info = gameState.evaluation.dynamic_info;
            aiStatusTextEl.innerHTML = `AI 落子完成 · <span style="color:var(--nord-yellow);font-weight:600;">${info.tag}</span>`;
        }
        const humanColor = (gameState && gameState.player_color) || 1;
        onMovePlacedClock(-humanColor);
    } catch (error) {
        console.error(error);
        aiStatusTextEl.innerText = "AI 响应异常";
    } finally {
        isAiThinking = false;
        aiSpinnerEl.style.display = "none";
        updateUI();
    }
}

async function makeMove(r, c) {
    try {
        // Clear candidates immediately upon human move click
        if (gameState && gameState.evaluation) {
            gameState.evaluation.top_candidates = [];
        }
        hoveredCandidatePoint = null;
        updateUI();

        const response = await fetch(`${API_URL}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ r, c, is_pass: false })
        });

        if (!response.ok) {
            return;
        }

        gameState = await response.json();
        // Keep candidates hidden while waiting for AI response
        if (gameState && gameState.evaluation) {
            gameState.evaluation.top_candidates = [];
        }
        playStoneSound();
        const humanColor = (gameState && gameState.player_color) || 1;
        onMovePlacedClock(humanColor);
        if (gameState.evaluation) {
            winrateHistory.push(gameState.evaluation.black_winrate);
        }
        updateUI();

        // Auto AI response if not game over
        if (!gameState.is_game_over && gameState.current_player !== humanColor) {
            setTimeout(() => {
                triggerAI();
            }, 100);
        }
    } catch (error) {
        console.error(error);
    }
}

// SGF Export Function
function exportSGF() {
    if (!gameState) return;

    const size = getBoardSize();
    const dateStr = new Date().toISOString().split('T')[0];
    const komiVal = gameState.komi !== undefined ? gameState.komi : 7.0;
    let sgf = `(;GM[1]FF[4]CA[UTF-8]AP[ZenGoAI:v2.0]SZ[${size}]KM[${komiVal}]RU[Chinese]PW[KataGo/AI]PB[Human]DT[${dateStr}]`;

    const toSgfCoord = (r, c) => {
        const colChar = String.fromCharCode(97 + c);
        const rowChar = String.fromCharCode(97 + r);
        return `${colChar}${rowChar}`;
    };

    if (gameState.move_history && gameState.move_history.length > 0) {
        gameState.move_history.forEach(([color, action]) => {
            const playerTag = color === 1 ? 'B' : 'W';
            if (action === 'pass') {
                sgf += `;${playerTag}[]`;
            } else {
                sgf += `;${playerTag}[${toSgfCoord(action[0], action[1])}]`;
            }
        });
    }

    sgf += `)`;

    const blob = new Blob([sgf], { type: 'application/x-go-sgf;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ZenGo_${size}x${size}_${Date.now()}.sgf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Controls Listeners
document.getElementById('btn-pass').addEventListener('click', async () => {
    try {
        if (gameState && gameState.evaluation) {
            gameState.evaluation.top_candidates = [];
        }
        hoveredCandidatePoint = null;
        updateUI();

        const response = await fetch(`${API_URL}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_pass: true })
        });
        gameState = await response.json();
        if (gameState && gameState.evaluation) {
            gameState.evaluation.top_candidates = [];
        }
        playStoneSound();
        const humanColor = (gameState && gameState.player_color) || 1;
        onMovePlacedClock(humanColor);
        if (gameState.evaluation) {
            winrateHistory.push(gameState.evaluation.black_winrate);
        }
        updateUI();

        if (!gameState.is_game_over && gameState.current_player !== humanColor) {
            setTimeout(() => {
                triggerAI();
            }, 100);
        }
    } catch (error) {
        console.error(error);
    }
});

document.getElementById('btn-undo').addEventListener('click', async () => {
    try {
        const response = await fetch(`${API_URL}/undo`, { method: 'POST' });
        gameState = await response.json();
        if (winrateHistory.length > 2) {
            winrateHistory.splice(-2);
        } else {
            winrateHistory = [gameState.evaluation ? gameState.evaluation.black_winrate : 50.0];
        }
        updateUI();
    } catch (error) {
        console.error(error);
    }
});

document.getElementById('btn-resign').addEventListener('click', async () => {
    if (!confirm("确定要认输吗？")) return;
    try {
        const response = await fetch(`${API_URL}/resign`, { method: 'POST' });
        gameState = await response.json();
        updateUI();
    } catch (error) {
        console.error(error);
    }
});

document.getElementById('btn-export-sgf').addEventListener('click', () => {
    exportSGF();
});

document.getElementById('btn-reset').addEventListener('click', async () => {
    if (!confirm("确定要清空盘面并重新开始一局吗？")) return;
    try {
        userDeclinedScoringMove = -1;
        if (aiScoringModalEl) aiScoringModalEl.style.display = 'none';
        const response = await fetch(`${API_URL}/reset`, { method: 'POST' });
        gameState = await response.json();
        winrateHistory = [gameState.evaluation ? gameState.evaluation.black_winrate : 50.0];
        initClock();
        updateUI();
    } catch (error) {
        console.error(error);
    }
});

// AI 提议终局数子弹窗交互监听
if (btnScoringAgreeEl) {
    btnScoringAgreeEl.addEventListener('click', async () => {
        if (!gameState || gameState.is_game_over) return;
        if (aiScoringModalEl) aiScoringModalEl.style.display = 'none';
        try {
            const response = await fetch(`${API_URL}/request_scoring`, { method: 'POST' });
            gameState = await response.json();
            showTerritory = true;
            if (toggleTerritoryEl) toggleTerritoryEl.checked = true;
            if (territoryCardEl) territoryCardEl.style.display = 'block';
            updateUI();
        } catch (err) {
            console.error(err);
        }
    });
}

if (btnScoringDeclineEl) {
    btnScoringDeclineEl.addEventListener('click', () => {
        if (aiScoringModalEl) aiScoringModalEl.style.display = 'none';
        userDeclinedScoringMove = gameState ? gameState.move_count : -1;
        aiStatusTextEl.innerText = "玩家暂不数子，请继续落子 (若官子收完 AI 将再次提议)...";
    });
}

// Request Final Area Scoring (申请数子判胜)
const btnRequestScoringEl = document.getElementById('btn-request-scoring');
if (btnRequestScoringEl) {
    btnRequestScoringEl.addEventListener('click', async () => {
        if (!gameState) return;
        if (gameState.is_game_over) {
            alert("当前对局已经结束！");
            return;
        }
        if (!confirm("是否向 AI 裁判申请终局数子并立即裁定胜负？\n(裁判将根据盘面活子与形势判断精确裁决)")) return;
        try {
            const response = await fetch(`${API_URL}/request_scoring`, { method: 'POST' });
            gameState = await response.json();
            showTerritory = true;
            if (toggleTerritoryEl) toggleTerritoryEl.checked = true;
            if (territoryCardEl) territoryCardEl.style.display = 'block';
            updateUI();
        } catch (err) {
            console.error(err);
        }
    });
}

// 实时形势判断 (目数卡片) 控制
const territoryCardEl = document.getElementById('territory-card');
const btnCloseTerrEl = document.getElementById('btn-close-terr');

if (btnCloseTerrEl) {
    btnCloseTerrEl.addEventListener('click', () => {
        showTerritory = false;
        if (toggleTerritoryEl) toggleTerritoryEl.checked = false;
        if (territoryCardEl) territoryCardEl.style.display = 'none';
        drawBoard();
    });
}

const selectResignModeEl = document.getElementById('select-resign-mode');
if (selectResignModeEl) {
    selectResignModeEl.addEventListener('change', async (e) => {
        try {
            await fetch(`${API_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ resign_mode: e.target.value })
            });
        } catch (err) {
            console.error(err);
        }
    });
}

const selectBoardSizeEl = document.getElementById('select-board-size');
if (selectBoardSizeEl) {
    selectBoardSizeEl.addEventListener('change', async (e) => {
        const size = parseInt(e.target.value) || 9;
        try {
            await fetch(`${API_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ board_size: size })
            });
            const res = await fetch(`${API_URL}/state`);
            gameState = await res.json();
            winrateHistory = [gameState.evaluation ? gameState.evaluation.black_winrate : 50.0];
            
            const optAz = selectOpponentEngineEl ? selectOpponentEngineEl.querySelector('option[value="alphazero"]') : null;
            if (optAz) optAz.disabled = (size === 19);
            if (size === 19 && selectOpponentEngineEl) selectOpponentEngineEl.value = "katago";
            
            const optRefAz = selectRefereeEngineEl ? selectRefereeEngineEl.querySelector('option[value="alphazero"]') : null;
            if (optRefAz) optRefAz.disabled = (size === 19);
            if (size === 19 && selectRefereeEngineEl) selectRefereeEngineEl.value = "katago";

            updateUI();
        } catch (err) {
            console.error(err);
        }
    });
}

const selectOpponentEngineEl = document.getElementById('select-opponent-engine');
if (selectOpponentEngineEl) {
    selectOpponentEngineEl.addEventListener('change', async (e) => {
        try {
            await fetch(`${API_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ opponent_engine: e.target.value })
            });
        } catch (err) {
            console.error(err);
        }
    });
}

const selectRefereeEngineEl = document.getElementById('select-referee-engine');
if (selectRefereeEngineEl) {
    selectRefereeEngineEl.addEventListener('change', async (e) => {
        try {
            await fetch(`${API_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ referee_engine: e.target.value })
            });
            fetchState();
        } catch (err) {
            console.error(err);
        }
    });
}

const selectAiVisitsEl = document.getElementById('select-ai-visits');
if (selectAiVisitsEl) {
    selectAiVisitsEl.addEventListener('change', async (e) => {
        const val = parseInt(e.target.value);
        const visits = isNaN(val) ? -1 : val;
        try {
            await fetch(`${API_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ai_visits: visits })
            });
            fetchState();
        } catch (err) {
            console.error(err);
        }
    });
}

const selectKomiEl = document.getElementById('select-komi');
if (selectKomiEl) {
    selectKomiEl.addEventListener('change', async (e) => {
        try {
            await fetch(`${API_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ komi: parseFloat(e.target.value) })
            });
            const res = await fetch(`${API_URL}/state`);
            gameState = await res.json();
            if (gameState.evaluation) {
                winrateHistory = [gameState.evaluation.black_winrate];
            }
            updateUI();
        } catch (err) {
            console.error(err);
        }
    });
}

const selectGameSpecEl = document.getElementById('select-game-spec');
if (selectGameSpecEl) {
    selectGameSpecEl.addEventListener('change', async (e) => {
        const val = e.target.value;
        const parts = val.split('_');
        const colorMode = parts[0];
        const handicap = parseInt(parts[1]) || 0;
        const defaultKomi = handicap >= 1 ? 0.5 : 7.0;

        try {
            await fetch(`${API_URL}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    color_mode: colorMode,
                    handicap: handicap,
                    komi: defaultKomi
                })
            });
            const res = await fetch(`${API_URL}/reset`, { method: 'POST' });
            gameState = await res.json();
            winrateHistory = [gameState.evaluation ? gameState.evaluation.black_winrate : 50.0];
            const selectKomiEl = document.getElementById('select-komi');
            if (selectKomiEl) selectKomiEl.value = defaultKomi.toFixed(1);
            updateUI();
        } catch (err) {
            console.error(err);
        }
    });
}

const selectTimeControlEl = document.getElementById('select-time-control');
if (selectTimeControlEl) {
    selectTimeControlEl.addEventListener('change', (e) => {
        initClock(e.target.value);
    });
}

// Start & Init Clock
initClock();
fetchState();
