# Load and run the model:
import subprocess
import time
import os

# Start vllm server in the background
vllm_process = subprocess.Popen([
    'vllm',
    'serve',  # Subcommand must follow vllm
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    '--trust-remote-code',
    '--dtype', 'half',
    '--max-model-len', '16384', # This is max token input and output that you send and retrieve
    '--enable-chunked-prefill', 'true',
    '--tensor-parallel-size', '1'
], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

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


try:
    success, stdout, stderr = monitor_vllm_process(vllm_process)

    if not success:
        print("\n❌ VLLM server failed to start!")
        print("\nFull STDOUT:", stdout)
        print("\nFull STDERR:", stderr)
        sys.exit(1)

except KeyboardInterrupt:
    print("\n⚠️ Monitoring interrupted by user")
    # This should just exited the process of probing, not the vllm, if you want it then you coul uncomment this.
    vllm_process.terminate()
    try:
         vllm_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
         vllm_process.kill()

    stdout, stderr = vllm_process.communicate()
    if stdout: print("\nFinal STDOUT:", stdout.decode('utf-8'))
    if stderr: print("\nFinal STDERR:", stderr.decode('utf-8'))
    sys.exit(0)