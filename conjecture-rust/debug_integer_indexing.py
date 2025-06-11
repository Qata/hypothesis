#!/usr/bin/env python3

def python_zigzag_index(value, shrink_towards):
    """Python's exact zigzag indexing algorithm"""
    index = 2 * abs(shrink_towards - value)
    if value > shrink_towards:
        index -= 1
    return index

def python_zigzag_value(index, shrink_towards):
    """Python's exact zigzag value algorithm"""
    n = (index + 1) // 2
    if (index % 2) == 0:
        n *= -1
    return shrink_towards + n

def test_shrink_towards_1():
    """Test the failing case with shrink_towards=1"""
    shrink_towards = 1
    
    print(f"Testing with shrink_towards={shrink_towards}")
    print()
    
    # Generate the actual Python sequence for indices 0-10
    print("What Python's algorithm actually produces:")
    python_sequence = []
    for index in range(10):
        value = python_zigzag_value(index, shrink_towards)
        python_sequence.append(value)
        print(f"Index {index} -> Value {value:2d}")
    
    print(f"\nPython sequence: {python_sequence}")
    
    # Now verify the forward direction  
    print("\nVerifying forward direction (value -> index):")
    for value in range(-5, 6):
        index = python_zigzag_index(value, shrink_towards)
        print(f"Value {value:2d} -> Index {index}")
    
    print("\nTesting bounded range [-3, 3] with shrink_towards=1:")
    bounded_values = []
    for index in range(7):  # Should cover range [-3, 3]
        value = python_zigzag_value(index, shrink_towards)
        if -3 <= value <= 3:
            bounded_values.append(value)
            print(f"Index {index} -> Value {value:2d} (valid)")
        else:
            print(f"Index {index} -> Value {value:2d} (out of bounds)")
    
    print(f"\nActual bounded sequence: {bounded_values}")

if __name__ == "__main__":
    test_shrink_towards_1()