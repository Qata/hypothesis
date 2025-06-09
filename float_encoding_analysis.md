# Python Hypothesis Float Lexicographic Encoding Analysis

## Summary

This document analyzes how Python Hypothesis calculates the lexicographic encoding for floating-point numbers, specifically focusing on the value 2.5.

## Key Result

**For f = 2.5, the lexicographic encoding is: `9227875636482146305` (0x8010000000000001)**

## Algorithm Overview

Python Hypothesis uses a tagged union approach with two encodings:

1. **Simple encoding (tag bit = 0)**: For integers that fit in 56 bits, just use the integer value
2. **Complex encoding (tag bit = 1)**: For other floats, use a specialized lexicographic encoding

Since 2.5 is not an integer, it uses the complex encoding.

## Detailed Steps for 2.5

### 1. IEEE 754 Representation
```
2.5 = 0x4004000000000000
    = 0100000000000100000000000000000000000000000000000000000000000000
    = 1.01 * 2^1 in binary

Breakdown:
- Sign: 0 (positive)
- Exponent: 1024 (raw) = 1 (unbiased = 1024 - 1023)
- Mantissa: 0x4000000000000 = fractional part .01 in binary
```

### 2. Mantissa Update Process

For unbiased exponent = 1 (which is in range [1, 51]):
- Calculate fractional bits to reverse: `n_fractional_bits = 52 - 1 = 51`
- Extract fractional part: `0x4000000000000` (the .01 part)
- Reverse the 51 bits: `0x4000000000000` becomes `0x1`
- This optimization prefers smaller fractional parts for better shrinking

### 3. Exponent Encoding

Python uses a custom exponent ordering for better lexicographic properties:
- Positive exponents first (0, 1, 2, 3, ...)
- Then negative exponents in decreasing order (..., -2, -1)
- Exponent 1024 (unbiased 1) maps to encoding index 1

### 4. Final Assembly

```
Result = (tag_bit=1 << 63) | (encoded_exponent=1 << 52) | updated_mantissa=1
       = 0x8000000000000000 | 0x0010000000000000 | 0x0000000000000001
       = 0x8010000000000001
       = 9227875636482146305
```

## Key Python Code Locations

The implementation is found in:
- `/hypothesis-python/src/hypothesis/internal/conjecture/floats.py` - Main encoding logic
- `/hypothesis-python/src/hypothesis/internal/floats.py` - IEEE 754 bit manipulation utilities

### Core Functions

1. **`float_to_lex(f)`** - Main entry point, checks if simple then calls base_float_to_lex
2. **`base_float_to_lex(f)`** - Complex encoding path
3. **`update_mantissa(unbiased_exponent, mantissa)`** - Reverses fractional bits for better shrinking
4. **`encode_exponent(e)`** - Maps IEEE exponent to lexicographic order
5. **`is_simple(f)`** - Checks if float is an integer ≤ 56 bits

## Exponent Encoding Table (First Few Entries)

```
Index -> IEEE Exponent (Unbiased)
   0  ->     1023  (0)
   1  ->     1024  (1)    ← 2.5 uses this
   2  ->     1025  (2)
   3  ->     1026  (3)
   ...
1024  ->     1022  (-1)
1025  ->     1021  (-2)
1026  ->     1020  (-3)
```

## Comparison with Other Values

```
Value   Simple? Lex Encoding           Hex
2.0     True    2                      0x0000000000000002
2.5     False   9227875636482146305    0x8010000000000001
3.0     True    3                      0x0000000000000003
1.5     False   9223372036854775809    0x8000000000000001
```

Notice how:
- Integer values use simple encoding (small numbers)
- Non-integers use complex encoding (large numbers with tag bit set)
- This ensures integers shrink better than fractional values

## Key Insights

1. **Shrinking Priority**: Integers are always preferred over non-integers
2. **Fractional Optimization**: Bit reversal ensures smaller fractional parts are preferred
3. **Exponent Ordering**: Positive exponents come before negative ones
4. **Tagged Union**: The high bit distinguishes between simple and complex encodings

This encoding ensures that lexicographically smaller values represent "simpler" floating-point numbers for property-based testing.