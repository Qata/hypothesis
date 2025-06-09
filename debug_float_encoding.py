#!/usr/bin/env python3

import struct
import sys
import os

# Add the hypothesis source directory to the path
sys.path.insert(0, '/Users/ch/Develop/hyptothesis-fresh/hypothesis-python/src')

from hypothesis.internal.conjecture import floats as flt
from hypothesis.internal.floats import float_to_int, int_to_float

def debug_float_to_lex(f):
    print(f"\n=== Debugging float_to_lex({f}) ===")
    
    # Check if it's simple first
    is_simple = flt.is_simple(f)
    print(f"is_simple({f}) = {is_simple}")
    
    if is_simple:
        result = int(f)
        print(f"Simple path: int({f}) = {result}")
        return result
    
    print("Taking complex path via base_float_to_lex...")
    return debug_base_float_to_lex(f)

def debug_base_float_to_lex(f):
    print(f"\n--- base_float_to_lex({f}) ---")
    
    # Step 1: Convert float to int representation
    i = float_to_int(f)
    print(f"float_to_int({f}) = {i} = 0x{i:016x} = 0b{i:064b}")
    
    # Step 2: Mask off sign bit
    i &= (1 << 63) - 1
    print(f"After masking sign bit: {i} = 0x{i:016x} = 0b{i:064b}")
    
    # Step 3: Extract exponent and mantissa
    exponent = i >> 52
    mantissa = i & flt.MANTISSA_MASK
    print(f"Raw exponent: {exponent} = 0x{exponent:03x} = 0b{exponent:011b}")
    print(f"Raw mantissa: {mantissa} = 0x{mantissa:013x} = 0b{mantissa:052b}")
    
    # Show unbiased exponent
    unbiased_exponent = exponent - flt.BIAS
    print(f"Unbiased exponent: {exponent} - {flt.BIAS} = {unbiased_exponent}")
    
    # Step 4: Update mantissa
    original_mantissa = mantissa
    mantissa = flt.update_mantissa(unbiased_exponent, mantissa)
    print(f"update_mantissa({unbiased_exponent}, {original_mantissa}) = {mantissa}")
    print(f"Updated mantissa: {mantissa} = 0x{mantissa:013x} = 0b{mantissa:052b}")
    
    # Step 5: Encode exponent
    original_exponent = exponent
    exponent = flt.encode_exponent(exponent)
    print(f"encode_exponent({original_exponent}) = {exponent}")
    print(f"Encoded exponent: {exponent} = 0x{exponent:03x} = 0b{exponent:011b}")
    
    # Step 6: Combine with tag bit
    result = (1 << 63) | (exponent << 52) | mantissa
    print(f"Final result: (1 << 63) | ({exponent} << 52) | {mantissa}")
    print(f"             = 0x8000000000000000 | 0x{exponent << 52:016x} | 0x{mantissa:013x}")
    print(f"             = {result} = 0x{result:016x}")
    
    return result

def debug_update_mantissa(unbiased_exponent, mantissa):
    print(f"\n--- update_mantissa({unbiased_exponent}, {mantissa}) ---")
    print(f"Input mantissa: {mantissa} = 0x{mantissa:013x} = 0b{mantissa:052b}")
    
    if unbiased_exponent <= 0:
        print(f"unbiased_exponent ({unbiased_exponent}) <= 0, reversing all 52 bits")
        result = flt.reverse_bits(mantissa, 52)
        print(f"reverse_bits({mantissa}, 52) = {result}")
        print(f"Result: {result} = 0x{result:013x} = 0b{result:052b}")
    elif unbiased_exponent <= 51:
        print(f"unbiased_exponent ({unbiased_exponent}) in range [1, 51]")
        n_fractional_bits = 52 - unbiased_exponent
        print(f"n_fractional_bits = 52 - {unbiased_exponent} = {n_fractional_bits}")
        
        fractional_part = mantissa & ((1 << n_fractional_bits) - 1)
        print(f"fractional_part = {mantissa} & {(1 << n_fractional_bits) - 1} = {fractional_part}")
        print(f"fractional_part: {fractional_part} = 0x{fractional_part:x} = 0b{fractional_part:0{n_fractional_bits}b}")
        
        # Remove fractional part
        mantissa_without_frac = mantissa ^ fractional_part
        print(f"mantissa without fractional part: {mantissa} ^ {fractional_part} = {mantissa_without_frac}")
        
        # Reverse fractional part
        reversed_frac = flt.reverse_bits(fractional_part, n_fractional_bits)
        print(f"reverse_bits({fractional_part}, {n_fractional_bits}) = {reversed_frac}")
        print(f"reversed fractional part: {reversed_frac} = 0x{reversed_frac:x} = 0b{reversed_frac:0{n_fractional_bits}b}")
        
        # Combine
        result = mantissa_without_frac | reversed_frac
        print(f"final mantissa: {mantissa_without_frac} | {reversed_frac} = {result}")
        print(f"Result: {result} = 0x{result:013x} = 0b{result:052b}")
    else:
        print(f"unbiased_exponent ({unbiased_exponent}) >= 52, leaving mantissa unchanged")
        result = mantissa
        print(f"Result: {result} = 0x{result:013x} = 0b{result:052b}")
    
    return result

def show_exponent_encoding_table():
    print("\n=== Exponent Encoding Table (first 20 entries) ===")
    print("Index -> Exponent")
    for i in range(min(20, len(flt.ENCODING_TABLE))):
        exp = flt.ENCODING_TABLE[i] 
        unbiased = exp - flt.BIAS if exp != flt.MAX_EXPONENT else "inf"
        print(f"{i:2d} -> {exp:4d} (unbiased: {unbiased})")
    
    print("\n=== Decoding Table (entries around 1024) ===")
    print("Exponent -> Index")
    for exp in range(1020, 1028):
        if exp <= flt.MAX_EXPONENT:
            idx = flt.DECODING_TABLE[exp]
            unbiased = exp - flt.BIAS
            print(f"{exp:4d} (unbiased: {unbiased:2d}) -> {idx:2d}")

if __name__ == "__main__":
    # Show the encoding tables
    show_exponent_encoding_table()
    
    # Debug 2.5 specifically
    f = 2.5
    print(f"\n" + "="*60)
    print(f"ANALYZING {f}")
    print("="*60)
    
    # Show raw IEEE 754 representation
    raw_bits = float_to_int(f)
    print(f"IEEE 754 bits of {f}: {raw_bits} = 0x{raw_bits:016x} = 0b{raw_bits:064b}")
    
    # Break down IEEE 754
    sign = (raw_bits >> 63) & 1
    exponent = (raw_bits >> 52) & 0x7FF
    mantissa = raw_bits & 0xFFFFFFFFFFFFF
    print(f"IEEE 754 breakdown:")
    print(f"  Sign: {sign}")
    print(f"  Exponent: {exponent} (unbiased: {exponent - 1023})")
    print(f"  Mantissa: {mantissa} = 0x{mantissa:013x}")
    
    # Show the lexicographic encoding step by step
    lex_result = debug_float_to_lex(f)
    print(f"\nFinal lexicographic encoding of {f}: {lex_result}")
    print(f"In hex: 0x{lex_result:016x}")
    print(f"In binary: 0b{lex_result:064b}")
    
    # Verify round-trip
    decoded = flt.lex_to_float(lex_result)
    print(f"\nRound-trip check: lex_to_float({lex_result}) = {decoded}")
    print(f"Original == Decoded: {f == decoded}")