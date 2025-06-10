#!/usr/bin/env python3
"""
Verification script to test our Rust implementation against Python Hypothesis.

This script runs key test cases using Python's actual choice.py functions
and outputs the expected results for our Rust implementation to verify against.
"""

import math
import sys
import os

# Add hypothesis-python to path
sys.path.insert(0, '/home/ch/Develop/hypothesis-conjecture-rust-enhancement/hypothesis-python/src')

from hypothesis.internal.conjecture.choice import (
    choice_to_index, 
    choice_from_index,
    choice_equal,
    choice_permitted,
    zigzag_index,
    zigzag_value
)
from hypothesis.internal.intervalsets import IntervalSet

def test_integer_indexing():
    """Test Python's integer choice indexing behavior"""
    print("=== PYTHON INTEGER INDEXING VERIFICATION ===")
    
    # Test cases that should match our Rust implementation exactly
    test_cases = [
        # Unbounded cases
        {"constraints": {"min_value": None, "max_value": None, "weights": None, "shrink_towards": 0}, 
         "values": [0, 1, -1, 2, -2, 3, -3, 5]},
         
        {"constraints": {"min_value": None, "max_value": None, "weights": None, "shrink_towards": 2}, 
         "values": [2, 3, 1, 4, 0, 5, -1, 6]},
         
        # Bounded cases  
        {"constraints": {"min_value": -3, "max_value": 3, "weights": None, "shrink_towards": 0}, 
         "values": [0, 1, -1, 2, -2, 3, -3]},
         
        {"constraints": {"min_value": -3, "max_value": 3, "weights": None, "shrink_towards": 1}, 
         "values": [1, 2, 0, 3, -1, -2, -3]},
         
        # Semi-bounded below
        {"constraints": {"min_value": 3, "max_value": None, "weights": None, "shrink_towards": 5}, 
         "values": [5, 6, 4, 7, 3, 8, 9, 10]},
         
        # Semi-bounded above  
        {"constraints": {"min_value": None, "max_value": 3, "weights": None, "shrink_towards": 1}, 
         "values": [1, 2, 0, 3, -1, -2, -3, -4]},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Case {i+1}: {case['constraints']} ---")
        constraints = case['constraints']
        
        for value in case['values']:
            index = choice_to_index(value, constraints)
            recovered = choice_from_index(index, "integer", constraints)
            print(f"Value {value:3} -> Index {index:3} -> Recovered {recovered:3} | Equal: {choice_equal(value, recovered)}")

def test_boolean_indexing():
    """Test Python's boolean choice indexing behavior"""
    print("\n\n=== PYTHON BOOLEAN INDEXING VERIFICATION ===")
    
    test_cases = [
        {"p": 0.0, "values": [False]},  # Only False allowed
        {"p": 1.0, "values": [True]},   # Only True allowed  
        {"p": 0.5, "values": [False, True]},  # Both allowed
    ]
    
    for case in test_cases:
        print(f"\n--- Boolean p={case['p']} ---")
        constraints = {"p": case['p']}
        
        for value in case['values']:
            if choice_permitted(value, constraints):
                index = choice_to_index(value, constraints)
                recovered = choice_from_index(index, "boolean", constraints)
                print(f"Value {value} -> Index {index} -> Recovered {recovered} | Equal: {choice_equal(value, recovered)}")
            else:
                print(f"Value {value} -> NOT PERMITTED")

def test_float_indexing():
    """Test Python's float choice indexing behavior"""
    print("\n\n=== PYTHON FLOAT INDEXING VERIFICATION ===")
    
    # Test key float values to verify our encoding matches
    test_values = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 
                   float('inf'), float('-inf'), float('nan')]
    
    constraints = {"min_value": float('-inf'), "max_value": float('inf'), 
                  "allow_nan": True, "smallest_nonzero_magnitude": 0.0}
    
    print("--- Float indexing ---")
    for value in test_values:
        if choice_permitted(value, constraints):
            index = choice_to_index(value, constraints)
            recovered = choice_from_index(index, "float", constraints)
            print(f"Value {value:8} -> Index {index:20} -> Recovered {recovered:8} | Equal: {choice_equal(value, recovered)}")

def test_string_indexing():
    """Test Python's string choice indexing behavior"""
    print("\n\n=== PYTHON STRING INDEXING VERIFICATION ===")
    
    # Simple alphabet case  
    intervals = IntervalSet.from_string("abc")
    constraints = {"min_size": 0, "max_size": 5, "intervals": intervals}
    
    test_values = ["", "a", "b", "c", "aa", "ab", "ba", "abc"]
    
    print("--- String indexing with alphabet 'abc' ---")
    for value in test_values:
        if choice_permitted(value, constraints):
            index = choice_to_index(value, constraints)
            recovered = choice_from_index(index, "string", constraints)
            print(f"Value '{value:3}' -> Index {index:3} -> Recovered '{recovered:3}' | Equal: {choice_equal(value, recovered)}")

def test_zigzag_functions():
    """Test Python's zigzag indexing functions directly"""
    print("\n\n=== PYTHON ZIGZAG FUNCTIONS VERIFICATION ===")
    
    # Test zigzag with shrink_towards=0
    print("--- Zigzag with shrink_towards=0 ---")
    for i in range(10):
        value = zigzag_value(i, shrink_towards=0)
        index = zigzag_index(value, shrink_towards=0)
        print(f"Index {i} -> Value {value:3} -> Index {index}")
        
    # Test zigzag with shrink_towards=2  
    print("\n--- Zigzag with shrink_towards=2 ---")
    for i in range(10):
        value = zigzag_value(i, shrink_towards=2)
        index = zigzag_index(value, shrink_towards=2)
        print(f"Index {i} -> Value {value:3} -> Index {index}")

def test_critical_edge_cases():
    """Test edge cases that often reveal implementation differences"""
    print("\n\n=== PYTHON CRITICAL EDGE CASES ===")
    
    # Integer edge cases
    print("--- Integer edge cases ---")
    edge_cases = [
        # Value exactly at shrink_towards
        (0, {"min_value": None, "max_value": None, "weights": None, "shrink_towards": 0}),
        (5, {"min_value": None, "max_value": None, "weights": None, "shrink_towards": 5}),
        
        # Boundary values in bounded ranges
        (-3, {"min_value": -3, "max_value": 3, "weights": None, "shrink_towards": 0}),
        (3, {"min_value": -3, "max_value": 3, "weights": None, "shrink_towards": 0}),
        
        # shrink_towards outside range (should be clamped)
        (1, {"min_value": 1, "max_value": 5, "weights": None, "shrink_towards": 0}),  # shrink_towards clamped to 1
        (5, {"min_value": 1, "max_value": 5, "weights": None, "shrink_towards": 10}), # shrink_towards clamped to 5
    ]
    
    for value, constraints in edge_cases:
        index = choice_to_index(value, constraints)
        recovered = choice_from_index(index, "integer", constraints)
        print(f"Value {value:3} with {constraints} -> Index {index} -> Recovered {recovered}")

if __name__ == "__main__":
    test_integer_indexing()
    test_boolean_indexing()
    test_float_indexing()
    test_string_indexing()
    test_zigzag_functions()
    test_critical_edge_cases()
    print("\n=== VERIFICATION COMPLETE ===")