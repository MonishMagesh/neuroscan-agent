import subprocess
import time
import sys

image = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ")

for attempt in range(1, 6):
    print(f"\nAttempt {attempt}/5...")
    result = subprocess.run(
        ["python", "cli.py", "analyze", image],
        capture_output=False
    )
    if result.returncode == 0:
        break
    print("Server busy — waiting 30 seconds...")
    time.sleep(30)