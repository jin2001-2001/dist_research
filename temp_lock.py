# temp_lock.py
# -*- coding: utf-8 -*-
# Ubuntu 22.04 仅标准库，两个函数：enter(id:int)、leave(id:int|None)
# 语义：id 越小优先（0 最高）；跨进程+跨线程；进程崩溃自动释放 flock

import os
import time
import errno
import fcntl
import threading

# === 配置 ===
WORKDIR  = os.path.abspath("./.pri_mutex")
RESOURCE = "global"
os.makedirs(WORKDIR, exist_ok=True)
LOCKFILE = os.path.join(WORKDIR, f"{RESOURCE}.lock")
WAITDIR  = os.path.join(WORKDIR, f"{RESOURCE}.waiters")
os.makedirs(WAITDIR, exist_ok=True)

# 进程内闸门：只在“真正尝试 flock”时短暂持有，避免同进程线程互相挤压队列
_proc_gate = threading.RLock()

# 线程局部状态
_tls = threading.local()

def _waitfile(pid: int, tid: int, idv: int) -> str:
    # 将 id 固定宽度零填充，便于解析；文件内容无意义，仅占位
    return os.path.join(WAITDIR, f"id{int(idv):020d}__{pid}_{tid}.w")

def _alive(pid: int) -> bool:
    # 进程是否存活（EPERM 也视为存活）
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM

def _parse_wait_name(name: str):
    # 从文件名解析 (id, pid, tid)
    # 形如 id000...123__4567_890.w
    try:
        if not (name.startswith("id") and "__" in name and name.endswith(".w")):
            return None
        id_str, rest = name[2:].split("__", 1)
        pid_str, tid_str = rest[:-2].split("_", 1)  # 去掉 .w
        return int(id_str), int(pid_str), int(tid_str)
    except Exception:
        return None

def enter(id: int):
    """
    进入点：阻塞直到获得互斥；id 越小优先（0 最高）。
    """
    if getattr(_tls, "holding", False):
        raise RuntimeError("already holding the mutex in this thread")

    pid, tid = os.getpid(), threading.get_ident()
    my_wait = _waitfile(pid, tid, id)

    # 创建自己的等待占位文件
    with open(my_wait, "wb") as f:
        f.write(b".")
        f.flush()
        os.fsync(f.fileno())

    # 打开全局互斥锁文件（flock crash-safe）
    lock_fd = os.open(LOCKFILE, os.O_CREAT | os.O_RDWR, 0o666)

    try:
        while True:
            # 扫描等待者：清理僵尸，找当前最小 id
            min_id = None
            try:
                for nm in os.listdir(WAITDIR):
                    parsed = _parse_wait_name(nm)
                    if not parsed:
                        continue
                    wid, wpid, _wtid = parsed
                    if _alive(wpid):
                        if (min_id is None) or (wid < min_id):
                            min_id = wid
                    else:
                        # 进程已死，清理残留
                        try:
                            os.remove(os.path.join(WAITDIR, nm))
                        except FileNotFoundError:
                            pass
                        except Exception:
                            pass
            except FileNotFoundError:
                os.makedirs(WAITDIR, exist_ok=True)

            # 若队列里存在更小 id，则直接小睡让路（避免无谓尝试）
            if (min_id is not None) and (id != min_id):
                time.sleep(0.002)
                continue

            # 轮到我：仅此时获取进程内闸门，再尝试 flock
            _proc_gate.acquire()
            try:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 成功进入临界区
                    _tls.holding = True
                    _tls.lock_fd = lock_fd
                    _tls.my_wait = my_wait
                    _tls.id = int(id)
                    return
                except OSError:
                    # 全局锁仍被他人持有：放开闸门，稍后再试
                    _proc_gate.release()
                    time.sleep(0.002)
                    continue
            except:
                # 异常时务必放开闸门
                try:
                    _proc_gate.release()
                except RuntimeError:
                    pass
                raise

    except:
        # 进入过程中出错：清理我的等待文件与 lock_fd
        try:
            os.remove(my_wait)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        try:
            os.close(lock_fd)
        except Exception:
            pass
        raise

def leave(id: int | None = None):
    """
    退出点：释放互斥；id 可为 None（仅用于一致性校验）。
    """
    if not getattr(_tls, "holding", False):
        raise RuntimeError("not holding the mutex in this thread")

    if id is not None and int(id) != getattr(_tls, "id", None):
        raise RuntimeError(f"leave(id={id}) != enter(id={getattr(_tls,'id',None)})")

    try:
        # 释放全局 flock
        fcntl.flock(_tls.lock_fd, fcntl.LOCK_UN)
    finally:
        try:
            os.close(_tls.lock_fd)
        except Exception:
            pass
        # 删除我的等待占位
        try:
            os.remove(_tls.my_wait)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        # 放开进程内闸门
        try:
            _proc_gate.release()
        except RuntimeError:
            pass
        # 清理线程局部状态
        _tls.holding = False
        _tls.lock_fd = None
        _tls.my_wait = None
        _tls.id = None


# --- 可选：自测 ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int)
    parser.add_argument("--hold", type=float, default=2.0)
    a = parser.parse_args()
    print(f"[pid={os.getpid()} tid={threading.get_ident()}] enter({a.id})")
    enter(a.id)
    print(f"[pid={os.getpid()}] ENTERED (id={a.id})")
    try:
        time.sleep(a.hold)
    finally:
        leave(a.id)
        print(f"[pid={os.getpid()}] LEFT (id={a.id})")
