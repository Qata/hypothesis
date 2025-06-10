#!/usr/bin/env python3
"""
Analysis of the character shrink ordering used by Python Hypothesis.
This replicates the logic from IntervalSet.char_in_shrink_order.
"""

def analyze_shrink_order():
    """
    Analyze the shrink order logic from IntervalSet.char_in_shrink_order.
    
    The key insight is that Hypothesis reorders characters to put digits and 
    uppercase letters first to get better shrinking behavior.
    """
    print("=== CHARACTER SHRINK ORDER ANALYSIS ===")
    
    # Example 1: Simple ASCII range
    print("\nExample 1: ASCII printable characters (32-126)")
    chars = [chr(i) for i in range(32, 127)]  # ASCII printable
    print(f"Original order: {repr(''.join(chars[:20]))}...")
    
    # Find special indices like Hypothesis does
    idx_of_zero = chars.index('0') if '0' in chars else -1
    idx_of_Z = chars.index('Z') if 'Z' in chars else -1
    
    print(f"Index of '0': {idx_of_zero}")
    print(f"Index of 'Z': {idx_of_Z}")
    
    if idx_of_zero >= 0 and idx_of_Z >= 0:
        # Simulate the shrink order rewriting
        reordered = []
        for i in range(len(chars)):
            if i <= idx_of_Z:
                n = idx_of_Z - idx_of_zero
                if i <= n:
                    # Map [0, n] to [idx_of_zero, idx_of_Z] 
                    actual_i = i + idx_of_zero
                else:
                    # Map [n+1, idx_of_Z] to [0, idx_of_zero-1] (reversed)
                    actual_i = idx_of_zero - (i - n)
            else:
                actual_i = i
            reordered.append(chars[actual_i])
        
        print(f"Shrink order: {repr(''.join(reordered[:20]))}...")
        
        # Show the mapping for key characters
        print("\nKey character mappings:")
        for i, char in enumerate(reordered[:20]):
            orig_i = chars.index(char)
            print(f"  Shrink index {i:2d}: '{char}' (was at index {orig_i:2d})")
    
    print("\nExample 2: Simple alphabet 'abcde012Z'")
    alphabet = "abcde012Z"
    chars = list(alphabet)
    
    idx_of_zero = chars.index('0') if '0' in chars else -1
    idx_of_Z = chars.index('Z') if 'Z' in chars else -1
    
    print(f"Original: {alphabet}")
    print(f"Index of '0': {idx_of_zero}")
    print(f"Index of 'Z': {idx_of_Z}")
    
    if idx_of_zero >= 0 and idx_of_Z >= 0:
        reordered = []
        for i in range(len(chars)):
            if i <= idx_of_Z:
                n = idx_of_Z - idx_of_zero
                if i <= n:
                    actual_i = i + idx_of_zero
                else:
                    actual_i = idx_of_zero - (i - n)
            else:
                actual_i = i
            reordered.append(chars[actual_i])
        
        print(f"Reordered: {''.join(reordered)}")
        
        # Show detailed mapping
        print("Detailed mapping:")
        for i, char in enumerate(reordered):
            orig_i = chars.index(char)
            print(f"  Shrink index {i}: '{char}' (original index {orig_i})")

if __name__ == "__main__":
    analyze_shrink_order()