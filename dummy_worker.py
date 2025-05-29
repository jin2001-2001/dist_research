import time
import sys

print(f"[Dummy Worker] argv: {sys.argv}")
print("[Dummy Worker] Simulating work for 10 seconds...")
time.sleep(10)
print("[Dummy Worker] Done.")
