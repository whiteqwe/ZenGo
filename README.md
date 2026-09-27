# ⚪⚫ ZenGo (禅棋 AI)
### 基于 29 通道战术特征工程、KataGo 知识蒸馏与轻量 MCTS 的高性能围棋 AI 系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 项目亮点与核心机理

ZenGo 是一个基于**多维战术特征工程、KataGo 教师模型知识蒸馏与轻量化蒙特卡洛树搜索（MCTS）**构建的高性能开源围棋 AI 系统。针对传统中小型深度学习围棋模型在死活判决、征子推演与端局收官自填眼上的三大经典痛点，本项目提出了多项针对性改进：

* 🧩 **29 通道战术级 BFS 特征工程（Tactical Feature Engineering）**：
  突破传统仅靠 2 通道黑白落子图的局限，设计了包含气数热力图（1/2/3/4+ 气连通块）、真假眼拓扑判决、征子（Ladder）推演、吃子与逃跑路径等 29 维空间状态张量，使卷积神经网络从第一层即可直接感知棋盘全局图连通性。
* 🎓 **KataGo 教师跨架构知识蒸馏（Teacher-Student Distillation）**：
  构建自动化对弈管道，以世界顶尖围棋 AI KataGo（40 Block / 512 Channel 巨型网络）作为教师模型，采用温度软目标交叉熵（Soft-target Cross-Entropy with $T=1.5$）优化策略头（Policy），规避了传统 Softmax 雅可比极值饱和区的梯度弥散；以平滑 L1 损失对齐价值头（Value），配合 EMA 0.995 动量平滑。
* ⚔️ **30% 死活题对抗样本注入（Tsumego Injection）**：
  在自对弈蒸馏训练集中强制注入 30% 边角严苛死活题样本，彻底消除了小型网络在终局自填眼送吃的灾难性失误。
* 🌲 **逆境 FPU 优化的轻量 MCTS 搜索**：
  基于 PUCT 树搜索算法，引入自适应 FPU（First Play Urgency）折减机制，在逆境处于劣势局面时节省 35% 的无效发散算力，兼顾搜索深度与实时交互响应。
* 🖥️ **全栈式 Web 现代化对战平台**：
  基于 FastAPI 与原生 HTML5 Canvas 打造极简交互界面，支持 9×9 与 13×13 棋盘、实时胜率波动曲线、目数领跑预估与 AI 候选着法热力图。

---

## 📂 项目架构

```
ZenGo/
├── backend/
│   ├── ai/
│   │   ├── agent.py            # AI 对弈统一调度接口与落子策略
│   │   ├── network.py          # 双头策略-价值网络架构 (Dual-headed ResNet)
│   │   ├── mcts_alpha.py       # 改进型 PUCT 蒙特卡洛树搜索
│   │   ├── katago_service.py   # KataGo GTP 与 Analysis 引擎通信服务
│   │   ├── best_model.pth      # 9x9 训练就绪最佳权重 (~40MB)
│   │   └── best_model_13x13.pth# 13x13 训练就绪最佳权重 (~42MB)
│   ├── engine/
│   │   └── board.py            # 29通道广度优先搜索(BFS)围棋规则引擎
│   ├── katago/                 # KataGo 运行配置与下载脚本
│   ├── train/
│   │   ├── distill_pipeline.py # KataGo 知识蒸馏主训练流
│   │   ├── tsumego_generator.py# 局部死活题生成器
│   │   └── health_guard.py     # 梯度健康监控与异常检测
│   └── main.py                 # FastAPI 异步后端服务
├── frontend/
│   ├── index.html              # Web 对弈界面
│   ├── style.css               # 样式设计
│   └── app.js                  # 棋盘交互与胜率渲染
├── experiment_report.pdf       # 11页出版级技术与实验研究报告
├── ZenGo_围棋AI模型与系统设计报告.md # 完整技术设计文档
├── setup_katago.py             # 自动化环境与 KataGo 依赖部署脚本
├── requirements.txt            # Python 依赖清单
└── start_web.bat               # Windows 一键启动脚本
```

---

## 🚀 快速开始

### 1. 环境准备
推荐使用 Python 3.10 或更高版本：
```bash
git clone https://github.com/whiteqwe/ZenGo.git
cd ZenGo

# 创建并激活虚拟环境 (可选)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装基础依赖
pip install -r requirements.txt
```

### 2. 启动 Web 对战界面
项目内置了已完成知识蒸馏的训练权重（`best_model.pth`），可直接启动体验：

* **Windows 一键启动**：双击运行根目录下的 `start_web.bat`
* **命令行启动**：
```bash
python start_web.py
```
启动成功后，浏览器访问：**`http://127.0.0.1:8000`** 即可展开人机对弈！

### 3. 可选：启用 KataGo 教师引擎辅助
若需进行深入局面分析或复现蒸馏训练流，可运行辅助部署脚本自动下载对应的 KataGo 二进制和官方模型：
```bash
python setup_katago.py
```

---

## 📊 实验与算法评估

详细实验推演、网络消融对比与死活题通过率曲线请查阅项目附带的 **[experiment_report.pdf](experiment_report.pdf)** 与 **[设计报告](ZenGo_围棋AI模型与系统设计报告.md)**。

| 模型配置 | 9×9 棋力预估 | 死活题成功率 | 征子识别准确率 | 单步思考耗时 (CPU) |
| :--- | :---: | :---: | :---: | :---: |
| 基础 2 通道 CNN | 15K | 41.2% | 38.6% | ~12 ms |
| **ZenGo (29通道 + 蒸馏)** | **2D ~ 3D** | **92.8%** | **94.5%** | **~25 ms** |
| ZenGo + 400 次 MCTS | 4D+ | 96.4% | 98.1% | ~180 ms |

---

## 📜 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。
