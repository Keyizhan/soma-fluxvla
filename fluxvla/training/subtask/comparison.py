"""Validate all resolved methods before launching a shared-stage1 comparison."""
import argparse
import json
import subprocess
import sys
from .config import load_config, check_comparison

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--configs',nargs='+',required=True)
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    configs=[load_config(path) for path in a.configs]
    check_comparison(configs)
    print(json.dumps(dict(status='COMPARISON_CONFIGS_MATCH',configs=configs),indent=2))
    if not a.check:
        for path in a.configs:
            subprocess.run([sys.executable,'-m','fluxvla.training.subtask.run','--config',path],check=True)

if __name__=='__main__':main()
