#!/usr/bin/env python3
"""
Demonstration of how Python Hypothesis handles the 65-bit float indexing problem.

The key insight: Python integers have arbitrary precision (they're BigInts),
so (sign << 64) | float_to_lex(abs(choice)) naturally works without special handling.
"""

import struct
import math

def float_to_int(value: float) -> int:
    """Convert float to its bit representation as integer"""
    return struct.unpack('!Q', struct.pack('!d', value))[0]

def int_to_float(value: int) -> float:
    """Convert integer bit representation back to float"""
    return struct.unpack('!d', struct.pack('!Q', value))[0]

def simple_float_to_lex(f: float) -> int:
    """Simplified version of Python's float_to_lex for demo"""
    # In real Python code this is much more sophisticated
    # This is just to show the 64-bit value part
    return float_to_int(abs(f)) & ((1 << 64) - 1)

def python_float_choice_to_index(choice: float) -> int:
    """Python's actual approach to float choice indexing"""
    sign = int(math.copysign(1.0, choice) < 0)
    lex_value = simple_float_to_lex(abs(choice))
    
    # THE KEY OPERATION: This creates a 65-bit value!
    result = (sign << 64) | lex_value
    
    print(f"  Sign: {sign}")
    print(f"  Lex value: {hex(lex_value)} ({lex_value.bit_length()} bits)")
    print(f"  Combined: {hex(result)} ({result.bit_length()} bits)")
    
    return result

def python_float_choice_from_index(index: int) -> tuple[int, int]:
    """Extract sign and lex value from 65-bit index"""
    sign = -1 if index >> 64 else 1
    lex_value = index & ((1 << 64) - 1)
    return sign, lex_value

def test_float_values():
    """Test various float values to show 65-bit behavior"""
    test_cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.7976931348623157e+308,   # max float
        -1.7976931348623157e+308,  # min float (most negative)
        float('inf'),
        float('-inf'),
        float('nan'),
    ]
    
    print("Testing Python's 65-bit float indexing approach:")
    print("=" * 60)
    
    for i, test_float in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_float}")
        
        try:
            index = python_float_choice_to_index(test_float)
            sign, lex = python_float_choice_from_index(index)
            
            print(f"  Index bit length: {index.bit_length()}")
            print(f"  Round trip - Sign: {sign}, Lex: {hex(lex)}")
            
            # Show that Python handles this naturally
            print(f"  Python int type: {type(index)}")
            print(f"  Can store in Python int: YES (arbitrary precision)")
            
        except Exception as e:
            print(f"  Error: {e}")

def test_max_cases():
    """Test the edge cases that create maximum 65-bit values"""
    print("\n" + "=" * 60)
    print("Testing maximum 65-bit cases:")
    
    # Case 1: sign=1, max lex value
    max_lex = (1 << 64) - 1
    max_65bit = (1 << 64) | max_lex
    
    print(f"\nMaximum 65-bit value:")
    print(f"  Sign=1, Lex=max: {hex(max_65bit)}")
    print(f"  Bit length: {max_65bit.bit_length()}")
    print(f"  Decimal: {max_65bit}")
    
    # Show extraction works
    extracted_sign = max_65bit >> 64
    extracted_lex = max_65bit & ((1 << 64) - 1)
    print(f"  Extracted sign: {extracted_sign}")
    print(f"  Extracted lex: {hex(extracted_lex)}")
    print(f"  Round trip: {extracted_sign == 1 and extracted_lex == max_lex}")

def demonstrate_key_insight():
    """Show the key insight about Python's solution"""
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Python's Solution to the 65-bit Problem")
    print("=" * 60)
    
    print("1. Python integers are arbitrary precision (BigInt)")
    print("2. No special handling needed for (sign << 64) | lex_value")
    print("3. The operation naturally creates 65-bit values when sign=1")
    print("4. Storage and retrieval work seamlessly with int type")
    
    # Demonstrate with actual large values
    print(f"\nPython sys.maxsize (native int size): {2**63-1}")
    print(f"Our 65-bit values can be: {(1 << 64) | ((1 << 64) - 1)}")
    print("Python handles both without issue!")
    
    print(f"\nComparison:")
    print(f"  64-bit signed max: {hex(2**63-1)} ({(2**63-1).bit_length()} bits)")
    print(f"  64-bit unsigned max: {hex(2**64-1)} ({(2**64-1).bit_length()} bits)")  
    print(f"  65-bit max: {hex((1 << 65)-1)} ({((1 << 65)-1).bit_length()} bits)")
    print("  Python int: All supported natively! 🎉")

if __name__ == "__main__":
    test_float_values()
    test_max_cases() 
    demonstrate_key_insight()