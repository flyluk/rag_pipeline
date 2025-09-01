import requests
import time
from typing import Tuple
import sys

def check_vllm_status(url: str = "http://localhost:8000/health") -> bool:
    """Check if VLLM server is running and healthy."""
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def monitor_vllm_process(vllm_process: subprocess.Popen, check_interval: int = 5) -> Tuple[bool, str, str]:
    """
    Monitor VLLM process and return status, stdout, and stderr.
    Returns: (success, stdout, stderr)
    """
    print("Starting VLLM server monitoring...")

    while vllm_process.poll() is None:  # While process is still running
        if check_vllm_status():
            print("✓ VLLM server is up and running!")
            return True, "", ""

        print("Waiting for VLLM server to start...")
        time.sleep(check_interval)

        # Check if there's any output to display
        if vllm_process.stdout.readable():
            stdout = vllm_process.stdout.read1().decode('utf-8')
            if stdout:
                print("STDOUT:", stdout)

        if vllm_process.stderr.readable():
            stderr = vllm_process.stderr.read1().decode('utf-8')
            if stderr:
                print("STDERR:", stderr)

    # If we get here, the process has ended
    stdout, stderr = vllm_process.communicate()
    return False, stdout.decode('utf-8'), stderr.decode('utf-8')