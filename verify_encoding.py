#!/usr/bin/env python3

import struct
from array import array

# Constants from the hypothesis code
MAX_EXPONENT = 0x7FF
BIAS = 1023
MANTISSA_MASK = (1 << 52) - 1

def float_to_int(value):
    return struct.unpack("!Q", struct.pack("!d", value))[0]

def exponent_key(e):
    if e == MAX_EXPONENT:
        return float("inf")
    unbiased = e - BIAS
    if unbiased < 0:
        return 10000 - unbiased
    else:
        return unbiased

# Build encoding tables
ENCODING_TABLE = array("H", sorted(range(MAX_EXPONENT + 1), key=exponent_key))
DECODING_TABLE = array("H", [0]) * len(ENCODING_TABLE)
for i, b in enumerate(ENCODING_TABLE):
    DECODING_TABLE[b] = i

def encode_exponent(e):
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

def float_to_lex(f):
    if is_simple(f):
        return int(f)
    
    i = float_to_int(f)
    i &= (1 << 63) - 1
    exponent = i >> 52
    mantissa = i & MANTISSA_MASK
    mantissa = update_mantissa(exponent - BIAS, mantissa)
    exponent = encode_exponent(exponent)
    return (1 << 63) | (exponent << 52) | mantissa

def test_values():
    test_cases = [
        ("2.0", 2.0),
        ("2.5", 2.5), 
        ("3.0", 3.0),
        ("1.5", 1.5),
        ("1.25", 1.25),
        ("1.75", 1.75),
        ("0.5", 0.5),
        ("0.25", 0.25),
        ("4.0", 4.0),
        ("8.0", 8.0),
    ]
    
    print("Value\t\tSimple?\tLex Encoding\t\tHex")
    print("-" * 60)
    
    for name, value in test_cases:
        simple = is_simple(value)
        lex = float_to_lex(value)
        print(f"{name:8s}\t{simple}\t{lex:20d}\t0x{lex:016x}")

def analyze_2_5_step_by_step():
    f = 2.5
    print(f"\n=== DETAILED ANALYSIS OF {f} ===")
    
    # IEEE 754 breakdown
    ieee_bits = float_to_int(f)
    print(f"IEEE 754 representation: 0x{ieee_bits:016x}")
    print(f"Binary: {ieee_bits:064b}")
    
    sign = (ieee_bits >> 63) & 1
    exponent = (ieee_bits >> 52) & 0x7FF
    mantissa = ieee_bits & ((1 << 52) - 1)
    
    print(f"\nIEEE 754 breakdown:")
    print(f"  Sign: {sign}")
    print(f"  Exponent: {exponent} (raw), {exponent - BIAS} (unbiased)")
    print(f"  Mantissa: {mantissa} = 0x{mantissa:013x}")
    print(f"  Binary mantissa: {mantissa:052b}")
    
    # The fractional part for 2.5
    # 2.5 = 10.1 in binary = 1.01 * 2^1
    # So mantissa represents .01 followed by zeros
    print(f"\nMantissa interpretation:")
    print(f"  2.5 = 1.01 * 2^1 in binary")
    print(f"  Mantissa represents the fractional part .01")
    print(f"  In 52-bit mantissa: 01 followed by 50 zeros")
    print(f"  0x{mantissa:013x} = {mantissa:052b}")
    
    # Update mantissa process
    unbiased_exp = exponent - BIAS
    print(f"\nMantissa update for unbiased exponent {unbiased_exp}:")
    print(f"  Since 1 <= {unbiased_exp} <= 51, we reverse the fractional bits")
    print(f"  n_fractional_bits = 52 - {unbiased_exp} = {52 - unbiased_exp}")
    
    n_frac = 52 - unbiased_exp
    frac_part = mantissa & ((1 << n_frac) - 1)
    print(f"  Fractional part: {frac_part} = 0x{frac_part:x}")
    print(f"  Binary fractional part: {frac_part:051b}")
    
    reversed_frac = reverse_bits(frac_part, n_frac)
    print(f"  Reversed fractional part: {reversed_frac} = 0x{reversed_frac:x}")
    print(f"  Binary reversed: {reversed_frac:051b}")
    
    # Final result
    updated_mantissa = (mantissa ^ frac_part) | reversed_frac
    encoded_exp = encode_exponent(exponent)
    
    print(f"\nFinal encoding:")
    print(f"  Updated mantissa: {updated_mantissa}")
    print(f"  Encoded exponent: {encoded_exp}")
    print(f"  Result: (1 << 63) | ({encoded_exp} << 52) | {updated_mantissa}")
    
    result = (1 << 63) | (encoded_exp << 52) | updated_mantissa
    print(f"  = {result} = 0x{result:016x}")

if __name__ == "__main__":
    test_values()
    analyze_2_5_step_by_step()