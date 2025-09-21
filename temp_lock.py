# pri_mutex_ubuntu.py
# 两个函数：enter(id:int) / leave(id:int|None)
# 规则：id 越大优先进入。支持多进程+多线程。仅依赖标准库。

import os, time, threading, errno
import fcntl

# 配置：如需多个独立资源，改 RESOURCE 名字并各放一份本文件即可
WORKDIR  = os.path.abspath("./.pri_mutex")
RESOURCE = "global"
os.makedirs(WORKDIR, exist_ok=True)
LOCKFILE = os.path.join(WORKDIR, f"{RESOURCE}.lock")
WAITDIR  = os.path.join(WORKDIR, f"{RESOURCE}.waiters")
os.makedirs(WAITDIR, exist_ok=True)

# 进程内线程互斥，避免 flock 在同一进程内的“可重入”效应
_proc_gate = threading.RLock()

# 线程局部状态（记录是否持有）
_tls = threading.local()

def _waitfile(pid: int, tid: int, idv: int) -> str:
    # 文件名中包含 id，便于扫描；零填充让字符串排序也正确
    return os.path.join(WAITDIR, f"id{int(idv):020d}__{pid}_{tid}.w")

def _alive(pid: int) -> bool:
    # 检测进程是否存在
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM  # 没权限也算活着

def _parse(name: str):
    # 从文件名提取 (id, pid, tid)
    # 形如 id000...123__4567_890.w
    try:
        if not (name.startswith("id") and "__" in name and name.endswith(".w")):
            return None
        id_str, rest = name[2:].split("__", 1)
        pid_str, tid_str = rest[:-2].split("_", 1)  # 去掉结尾 .w
        return int(id_str), int(pid_str), int(tid_str)
    except Exception:
        return None

def enter(id: int):
    """进入点：阻塞直到获得互斥；id 越大优先。"""
    if getattr(_tls, "holding", False):
        raise RuntimeError("already holding the mutex in this thread")

    pid, tid = os.getpid(), threading.get_ident()
    my_wait = _waitfile(pid, tid, id)

    with open(my_wait, "wb") as f:
        f.write(b".")  # 仅占位
        f.flush()
        os.fsync(f.fileno())

    # 进程内先序列化，避免同进程多线程同时闯入
    _proc_gate.acquire()

    # 打开全局锁文件（flock 在进程崩溃时会自动释放）
    lock_fd = os.open(LOCKFILE, os.O_CREAT | os.O_RDWR, 0o666)

    try:
        while True:
            # 清理僵尸等待者 & 找到当前最小 id
            min_id = None
            try:
                for nm in os.listdir(WAITDIR):
                    p = _parse(nm)
                    if not p:
                        continue
                    wid, wpid, _wtid = p
                    if _alive(wpid):
                        if (min_id is None) or (wid < min_id):
                            min_id = wid
                    else:
                        # 进程已死，清理残留
                        try: os.remove(os.path.join(WAITDIR, nm))
                        except FileNotFoundError: pass
                        except Exception: pass
            except FileNotFoundError:
                os.makedirs(WAITDIR, exist_ok=True)

            # 轮到我（我是当前最小 id），尝试拿全局互斥
            if (min_id is None) or (id >= min_id):
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 成功：进入临界区
                    _tls.holding = True
                    _tls.lock_fd = lock_fd
                    _tls.my_wait = my_wait
                    _tls.id = int(id)
                    return
                except OSError:
                    # 还被别人占用，稍等
                    pass

            time.sleep(0.002)  # 2ms 退避，低 CPU 占用

    except:
        # 进入过程中出错：释放进程内闸门 & 清理我的等待文件
        try: _proc_gate.release()
        except RuntimeError: pass
        try: os.remove(my_wait)
        except FileNotFoundError: pass
        except Exception: pass
        os.close(lock_fd)
        raise

def leave(id: int | None = None):
    """退出点：释放互斥；id 可不填（用于一致性校验）。"""
    if not getattr(_tls, "holding", False):
        raise RuntimeError("not holding the mutex in this thread")

    if id is not None and int(id) != getattr(_tls, "id", None):
        raise RuntimeError(f"leave(id={id}) != enter(id={getattr(_tls,'id',None)})")

    try:
        # 释放全局 flock
        fcntl.flock(_tls.lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(_tls.lock_fd)
        # 清理我的等待文件
        try: os.remove(_tls.my_wait)
        except FileNotFoundError: pass
        except Exception: pass
        # 释放进程内闸门
        try: _proc_gate.release()
        except RuntimeError: pass
        # 重置线程状态
        _tls.holding = False
        _tls.lock_fd = None
        _tls.my_wait = None
        _tls.id = None


# --- 简单自测：不同终端同时运行观察顺序（可删） ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int)
    parser.add_argument("--hold", type=float, default=3.0)
    a = parser.parse_args()
    print(f"[pid={os.getpid()} tid={threading.get_ident()}] enter({a.id}) ...")
    enter(a.id)
    print(f"[pid={os.getpid()}] ENTERED (id={a.id})")
    try:
        time.sleep(a.hold)
    finally:
        leave(a.id)
        print(f"[pid={os.getpid()}] LEFT (id={a.id})")
