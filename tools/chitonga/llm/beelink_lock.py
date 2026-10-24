# 23 Oct (Chris: 11 h of subscription work, six streams): ONE heavy Beelink job at a time. Use:
#   python beelink_lock.py acquire <name>   (waits until free, then holds the lock)   ...   python beelink_lock.py release <name>
# or in Python:  with beelink_lock.held("A-lessons"): ...
import contextlib, json, os, sys, time
LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".beelink_lock")


def acquire(name, poll=20, stale_h=4):
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"by": name, "at": time.time()}).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                d = json.load(open(LOCK))
                if time.time() - d.get("at", 0) > stale_h * 3600:
                    os.remove(LOCK)
                    continue
            except Exception:  # noqa: BLE001
                pass
            time.sleep(poll)


def release(name):
    try:
        if json.load(open(LOCK)).get("by") == name:
            os.remove(LOCK)
    except FileNotFoundError:
        pass


@contextlib.contextmanager
def held(name):
    acquire(name)
    try:
        yield
    finally:
        release(name)


if __name__ == "__main__":
    {"acquire": acquire, "release": release}[sys.argv[1]](sys.argv[2])
