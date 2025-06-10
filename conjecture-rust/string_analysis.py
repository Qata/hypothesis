#!/usr/bin/env python3
"""
Manual analysis of Python Hypothesis string choice indexing to understand the exact algorithm.
Based on the source code analysis.
"""

def _size_to_index(size, alphabet_size):
    """Closed form of geometric series: sum(alphabet_size**i for i in range(size))"""
    if alphabet_size <= 0:
        assert size == 0
        return 0
    if alphabet_size == 1:
        return size
    return (alphabet_size**size - 1) // (alphabet_size - 1)

def _index_to_size(index, alphabet_size):
    """Inverse of _size_to_index"""
    if alphabet_size == 0:
        return 0
    elif alphabet_size == 1:
        return index
    
    import math
    total = index * (alphabet_size - 1) + 1
    size = math.log(total, alphabet_size)
    
    # Handle floating point precision issues
    if 0 < math.ceil(size) - size < 1e-7:
        s = 0
        while total >= alphabet_size:
            total //= alphabet_size
            s += 1
        return s
    return math.floor(size)

def collection_index(choice, min_size, alphabet_size, to_order_func):
    """Convert a collection to an index based on size + content ordering"""
    # Start by adding the size to the index, relative to min_size
    index = _size_to_index(len(choice), alphabet_size) - _size_to_index(min_size, alphabet_size)
    
    # Add each element to the index, starting from the end
    running_exp = 1
    for c in reversed(choice):
        index += running_exp * to_order_func(c)
        running_exp *= alphabet_size
    return index

def collection_value(index, min_size, alphabet_size, from_order_func):
    """Convert an index back to a collection"""
    index += _size_to_index(min_size, alphabet_size)
    size = _index_to_size(index, alphabet_size)
    
    # Subtract out the amount responsible for the size
    index -= _size_to_index(size, alphabet_size)
    vals = []
    for i in reversed(range(size)):
        if index == 0:
            n = 0
        else:
            n = index // (alphabet_size**i)
            index -= n * (alphabet_size**i)
        vals.append(from_order_func(n))
    return vals

# Simulate IntervalSet.char_in_shrink_order for alphabet "abc"
def char_in_shrink_order(i, alphabet):
    """Simple version - just return characters in order for now"""
    return alphabet[i]

def index_from_char_in_shrink_order(c, alphabet):
    """Inverse of char_in_shrink_order"""
    return alphabet.index(c)

def test_string_indexing():
    print("=== STRING CHOICE INDEXING ANALYSIS ===")
    alphabet = "abc"
    alphabet_size = len(alphabet)
    min_size = 0
    max_size = 3
    
    print(f"Alphabet: {alphabet}")
    print(f"Alphabet size: {alphabet_size}")
    print(f"Min size: {min_size}, Max size: {max_size}")
    print()
    
    # Test size indexing first
    print("Size indexing:")
    for size in range(max_size + 1):
        size_index = _size_to_index(size, alphabet_size)
        print(f"  Size {size} -> index {size_index}")
    print()
    
    # Test string indexing
    print("String choice indexing:")
    for i in range(min(30, alphabet_size**(max_size+1))):
        try:
            string_chars = collection_value(
                i, min_size, alphabet_size, 
                lambda n: char_in_shrink_order(n, alphabet)
            )
            string = ''.join(string_chars)
            
            back_index = collection_index(
                string, min_size, alphabet_size,
                lambda c: index_from_char_in_shrink_order(c, alphabet)
            )
            
            print(f"  Index {i:2d}: '{string}' (len={len(string)}) -> back to index {back_index}")
            
            if back_index != i:
                print(f"    ERROR: Index mismatch!")
                break
                
        except Exception as e:
            print(f"  Index {i}: ERROR: {e}")
            break

if __name__ == "__main__":
    test_string_indexing()