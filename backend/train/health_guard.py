import os
import sys
import time
import subprocess
import psutil
import torch

def get_gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8', timeout=5
        ).strip()
        parts = [p.strip() for p in out.split(',')]
        return {
            "temp": float(parts[0]),
            "util": float(parts[1]),
            "power": float(parts[2]),
            "mem_used": float(parts[3]),
            "mem_total": float(parts[4])
        }
    except Exception:
        return None

def get_ram_telemetry():
    mem = psutil.virtual_memory()
    return {
        "percent": mem.percent,
        "used_gb": round(mem.used / (1024**3), 2),
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2)
    }

def clean_orphan_processes():
    killed_any = False
    for p in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            p_name = p.info['name'].lower()
            mem_mb = p.info['memory_info'].rss / (1024 * 1024)
            # Kill runaway KataGo processes exceeding 2.5 GB RAM
            if 'katago' in p_name and mem_mb > 2500:
                print(f"[HealthGuard] Detected bloated KataGo (PID {p.pid}, {mem_mb:.1f} MB), terminating to free RAM...")
                p.kill()
                killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed_any

def guard_loop():
    print("[HealthGuard] 自动智能温控与内存看门狗已启动！(每 60 秒轮询守护系统健康)")
    while True:
        try:
            ram = get_ram_telemetry()
            gpu = get_gpu_telemetry()
            
            # 1. 内存过高保护 (> 82%)
            if ram['percent'] > 82.0:
                print(f"[HealthGuard Alert] 内存占比偏高 ({ram['percent']}%, 已用 {ram['used_gb']}G)，执行智能清理...")
                clean_orphan_processes()
                
            # 2. GPU 高温预警 (> 81°C)
            if gpu and gpu['temp'] > 81.0:
                print(f"[HealthGuard Warning] GPU 温度过高 ({gpu['temp']}°C)，等待温控回落...")
                time.sleep(10)
                
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(60)

if __name__ == "__main__":
    guard_loop()
