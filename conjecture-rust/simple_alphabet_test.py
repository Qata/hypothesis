#!/usr/bin/env python3
"""
Test string indexing with simple alphabets that don't contain '0' or 'Z'.
"""

def analyze_simple_alphabet():
    print("=== SIMPLE ALPHABET ANALYSIS ===")
    
    # Test with alphabet that has no '0' or 'Z'
    alphabet = "abc"
    print(f"Alphabet: {alphabet}")
    
    # In this case, the shrink order logic should not modify anything
    # because there's no '0' or 'Z' to use as anchors
    
    # Simulate what IntervalSet would do:
    chars = list(alphabet)
    try:
        idx_of_zero = chars.index('0')
    except ValueError:
        idx_of_zero = None
        
    try:
        idx_of_Z = chars.index('Z')
    except ValueError:
        idx_of_Z = None
    
    print(f"Index of '0': {idx_of_zero}")
    print(f"Index of 'Z': {idx_of_Z}")
    
    # When there's no '0' or 'Z', the shrink order should be the same as regular order
    print("Expected behavior: no reordering, characters stay in alphabetical order")
    print(f"Shrink order should be: {alphabet}")
    
    print("\nTesting with alphabet that includes '0':")
    alphabet2 = "abc0"
    chars2 = list(alphabet2)
    
    try:
        idx_of_zero = chars2.index('0')
    except ValueError:
        idx_of_zero = None
        
    try:
        idx_of_Z = chars2.index('Z')
    except ValueError:
        idx_of_Z = None
        
    print(f"Alphabet: {alphabet2}")
    print(f"Index of '0': {idx_of_zero}")
    print(f"Index of 'Z': {idx_of_Z}")
    
    # With '0' but no 'Z', let's see what happens
    # Looking at the code: _idx_of_Z = min(self.index_above(ord("Z")), len(self) - 1)
    # This means _idx_of_Z would be len(self) - 1 = 3 in this case
    simulated_idx_of_Z = min(len(chars2), len(chars2) - 1)  # This would be 3
    print(f"Simulated _idx_of_Z: {simulated_idx_of_Z}")
    
    # Now apply the reordering logic
    if idx_of_zero is not None:
        reordered = []
        for i in range(len(chars2)):
            if i <= simulated_idx_of_Z:
                n = simulated_idx_of_Z - idx_of_zero
                if i <= n:
                    actual_i = i + idx_of_zero
                    if actual_i < len(chars2):
                        reordered.append(chars2[actual_i])
                    else:
                        reordered.append('?')  # Error case
                else:
                    actual_i = idx_of_zero - (i - n)
                    if actual_i >= 0:
                        reordered.append(chars2[actual_i])
                    else:
                        reordered.append('?')  # Error case
            else:
                reordered.append(chars2[i])
        
        print(f"Reordered: {''.join(reordered)}")

if __name__ == "__main__":
    analyze_simple_alphabet()