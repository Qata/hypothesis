#!/usr/bin/env python3
"""
Run specific test cases to verify Rust vs Python parity
"""

import subprocess
import json

def run_rust_test_case(test_case_name):
    """Run a specific Rust test and capture output"""
    try:
        result = subprocess.run([
            'cargo', 'test', '--lib', f'test_{test_case_name}', '--', '--nocapture'
        ], capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Test timed out", 1

def main():
    print("=== RUST vs PYTHON VERIFICATION COMPARISON ===")
    
    # Key cases to verify
    test_cases = [
        'python_zigzag_parity',
        'python_boolean_parity', 
        'python_bounded_integer_ordering',
        'python_float_sign_bit_logic',
        'edge_case_shrink_towards_clamping'
    ]
    
    for case in test_cases:
        print(f"\n--- Running Rust test: {case} ---")
        stdout, stderr, returncode = run_rust_test_case(case)
        
        if returncode == 0:
            print("✅ PASSED")
            # Extract key verification lines
            lines = stdout.split('\n')
            for line in lines:
                if 'VERIFICATION DEBUG:' in line and ('test passed' in line or 'Value' in line):
                    print(f"  {line.strip()}")
        else:
            print("❌ FAILED")
            print(f"  Return code: {returncode}")
            if stderr:
                print(f"  Error: {stderr}")

if __name__ == "__main__":
    main()