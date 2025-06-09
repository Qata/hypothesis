#!/usr/bin/env python3

import struct
from array import array

# Constants from the hypothesis code
MAX_EXPONENT = 0x7FF
BIAS = 1023
MAX_POSITIVE_EXPONENT = MAX_EXPONENT - 1 - BIAS
MANTISSA_MASK = (1 << 52) - 1

def float_to_int(value):
    """Convert float to its integer bit representation"""
    return struct.unpack("!Q", struct.pack("!d", value))[0]

def int_to_float(value):
    """Convert integer bit representation back to float"""
    return struct.unpack("!d", struct.pack("!Q", value))[0]

def exponent_key(e):
    if e == MAX_EXPONENT:
        return float("inf")
    unbiased = e - BIAS
    if unbiased < 0:
        return 10000 - unbiased
    else:
        return unbiased

# Build the encoding and decoding tables
ENCODING_TABLE = array("H", sorted(range(MAX_EXPONENT + 1), key=exponent_key))
DECODING_TABLE = array("H", [0]) * len(ENCODING_TABLE)

for i, b in enumerate(ENCODING_TABLE):
    DECODING_TABLE[b] = i

def decode_exponent(e):
    assert 0 <= e <= MAX_EXPONENT
    return ENCODING_TABLE[e]

def encode_exponent(e):
    assert 0 <= e <= MAX_EXPONENT
    return DECODING_TABLE[e]

def reverse_byte(b):
    result = 0
    for _ in range(8):
        result <<= 1
        result |= b & 1
        b >>= 1
    return result

REVERSE_BITS_TABLE = bytearray(map(reverse_byte, range(256)))

def reverse64(v):
    assert v.bit_length() <= 64
    return (
        (REVERSE_BITS_TABLE[(v >> 0) & 0xFF] << 56)
        | (REVERSE_BITS_TABLE[(v >> 8) & 0xFF] << 48)
        | (REVERSE_BITS_TABLE[(v >> 16) & 0xFF] << 40)
        | (REVERSE_BITS_TABLE[(v >> 24) & 0xFF] << 32)
        | (REVERSE_BITS_TABLE[(v >> 32) & 0xFF] << 24)
        | (REVERSE_BITS_TABLE[(v >> 40) & 0xFF] << 16)
        | (REVERSE_BITS_TABLE[(v >> 48) & 0xFF] << 8)
        | (REVERSE_BITS_TABLE[(v >> 56) & 0xFF] << 0)
    )

def reverse_bits(x, n):
    assert x.bit_length() <= n <= 64
    x = reverse64(x)
    x >>= 64 - n
    return x

def update_mantissa(unbiased_exponent, mantissa):
    if unbiased_exponent <= 0:
        mantissa = reverse_bits(mantissa, 52)
    elif unbiased_exponent <= 51:
        n_fractional_bits = 52 - unbiased_exponent
        fractional_part = mantissa & ((1 << n_fractional_bits) - 1)
        mantissa ^= fractional_part
        mantissa |= reverse_bits(fractional_part, n_fractional_bits)
    return mantissa

def is_simple(f):
    try:
        i = int(f)
    except (ValueError, OverflowError):
        return False
    if i != f:
        return False
    return i.bit_length() <= 56

def base_float_to_lex(f):
    i = float_to_int(f)
    i &= (1 << 63) - 1
    exponent = i >> 52
    mantissa = i & MANTISSA_MASK
    mantissa = update_mantissa(exponent - BIAS, mantissa)
    exponent = encode_exponent(exponent)

    assert mantissa.bit_length() <= 52
    return (1 << 63) | (exponent << 52) | mantissa

def float_to_lex(f):
    if is_simple(f):
        assert f >= 0
        return int(f)
    return base_float_to_lex(f)

def debug_float_to_lex(f):
    print(f"\n=== Debugging float_to_lex({f}) ===")
    
    # Check if it's simple first
    simple = is_simple(f)
    print(f"is_simple({f}) = {simple}")
    
    if simple:
        result = int(f)
        print(f"Simple path: int({f}) = {result}")
        return result
    
    print("Taking complex path via base_float_to_lex...")
    
    # Show raw IEEE 754 representation
    raw_bits = float_to_int(f)
    print(f"IEEE 754 bits of {f}: {raw_bits} = 0x{raw_bits:016x}")
    
    # Break down IEEE 754
    sign = (raw_bits >> 63) & 1
    ieee_exponent = (raw_bits >> 52) & 0x7FF
    ieee_mantissa = raw_bits & 0xFFFFFFFFFFFFF
    print(f"IEEE 754 breakdown:")
    print(f"  Sign: {sign}")
    print(f"  Exponent: {ieee_exponent} (unbiased: {ieee_exponent - 1023})")
    print(f"  Mantissa: {ieee_mantissa} = 0x{ieee_mantissa:013x}")
    
    # Now process through base_float_to_lex step by step
    print(f"\n--- base_float_to_lex({f}) ---")
    
    # Step 1: Convert float to int representation
    i = float_to_int(f)
    print(f"float_to_int({f}) = {i} = 0x{i:016x}")
    
    # Step 2: Mask off sign bit
    i &= (1 << 63) - 1
    print(f"After masking sign bit: {i} = 0x{i:016x}")
    
    # Step 3: Extract exponent and mantissa
    exponent = i >> 52
    mantissa = i & MANTISSA_MASK
    print(f"Raw exponent: {exponent} = 0x{exponent:03x}")
    print(f"Raw mantissa: {mantissa} = 0x{mantissa:013x}")
    
    # Show unbiased exponent
    unbiased_exponent = exponent - BIAS
    print(f"Unbiased exponent: {exponent} - {BIAS} = {unbiased_exponent}")
    
    # Step 4: Update mantissa
    print(f"\n--- update_mantissa({unbiased_exponent}, {mantissa}) ---")
    print(f"Input mantissa: {mantissa} = 0x{mantissa:013x}")
    
    original_mantissa = mantissa
    if unbiased_exponent <= 0:
        print(f"unbiased_exponent ({unbiased_exponent}) <= 0, reversing all 52 bits")
        mantissa = reverse_bits(mantissa, 52)
        print(f"reverse_bits({original_mantissa}, 52) = {mantissa} = 0x{mantissa:013x}")
    elif unbiased_exponent <= 51:
        print(f"unbiased_exponent ({unbiased_exponent}) in range [1, 51]")
        n_fractional_bits = 52 - unbiased_exponent
        print(f"n_fractional_bits = 52 - {unbiased_exponent} = {n_fractional_bits}")
        
        fractional_part = mantissa & ((1 << n_fractional_bits) - 1)
        print(f"fractional_part = {mantissa} & {(1 << n_fractional_bits) - 1} = {fractional_part} = 0x{fractional_part:x}")
        
        # Remove fractional part
        mantissa_without_frac = mantissa ^ fractional_part
        print(f"mantissa without fractional part: {mantissa} ^ {fractional_part} = {mantissa_without_frac} = 0x{mantissa_without_frac:x}")
        
        # Reverse fractional part
        reversed_frac = reverse_bits(fractional_part, n_fractional_bits)
        print(f"reverse_bits({fractional_part}, {n_fractional_bits}) = {reversed_frac} = 0x{reversed_frac:x}")
        
        # Combine
        mantissa = mantissa_without_frac | reversed_frac
        print(f"final mantissa: {mantissa_without_frac} | {reversed_frac} = {mantissa} = 0x{mantissa:013x}")
    else:
        print(f"unbiased_exponent ({unbiased_exponent}) >= 52, leaving mantissa unchanged")
    
    print(f"Updated mantissa: {mantissa} = 0x{mantissa:013x}")
    
    # Step 5: Encode exponent
    original_exponent = exponent
    exponent = encode_exponent(exponent)
    print(f"encode_exponent({original_exponent}) = {exponent} = 0x{exponent:03x}")
    
    # Step 6: Combine with tag bit
    result = (1 << 63) | (exponent << 52) | mantissa
    print(f"Final result: (1 << 63) | ({exponent} << 52) | {mantissa}")
    print(f"             = 0x8000000000000000 | 0x{exponent << 52:016x} | 0x{mantissa:013x}")
    print(f"             = {result} = 0x{result:016x}")
    
    return result

def show_exponent_encoding():
    print("=== Exponent Encoding Table (first 20 entries) ===")
    print("Index -> Exponent")
    for i in range(min(20, len(ENCODING_TABLE))):
        exp = ENCODING_TABLE[i] 
        unbiased = exp - BIAS if exp != MAX_EXPONENT else "inf"
        print(f"{i:2d} -> {exp:4d} (unbiased: {unbiased})")
    
    print("\n=== Decoding Table (entries around 1024) ===")
    print("Exponent -> Index")
    for exp in range(1020, 1028):
        if exp <= MAX_EXPONENT:
            idx = DECODING_TABLE[exp]
            unbiased = exp - BIAS
            print(f"{exp:4d} (unbiased: {unbiased:2d}) -> {idx:2d}")

if __name__ == "__main__":
    # Show the encoding tables
    show_exponent_encoding()
    
    # Debug 2.5 specifically
    f = 2.5
    print(f"\n" + "="*60)
    print(f"ANALYZING {f}")
    print("="*60)
    
    lex_result = debug_float_to_lex(f)
    print(f"\nFinal lexicographic encoding of {f}: {lex_result}")
    print(f"In hex: 0x{lex_result:016x}")