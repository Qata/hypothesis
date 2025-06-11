#!/usr/bin/env python3

import hashlib
import sys

def int_from_bytes(data):
    """Convert bytes to int (big-endian)"""
    return int.from_bytes(data, byteorder='big')

def calc_label_from_name(name: str) -> int:
    """Exact copy of the function from Python Hypothesis"""
    hashed = hashlib.sha384(name.encode()).digest()
    return int_from_bytes(hashed[:8])

# Test the specific input
input_name = "ir draw record"
result = calc_label_from_name(input_name)

print(f"Input: '{input_name}'")
print(f"Result: {result}")
print(f"Result (hex): {hex(result)}")
print(f"Rust expected: 9223372036854775807")
print(f"Rust expected (hex): {hex(9223372036854775807)}")
print(f"Match: {result == 9223372036854775807}")

# Let's also show the intermediate steps for debugging
hashed = hashlib.sha384(input_name.encode()).digest()
print(f"\nDebugging info:")
print(f"SHA384 hash (full): {hashed.hex()}")
print(f"First 8 bytes: {hashed[:8].hex()}")
print(f"First 8 bytes as int: {int_from_bytes(hashed[:8])}")