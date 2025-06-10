#!/usr/bin/env python3

# Test Python's integer handling of large numbers
val = (1 << 64) | 0xFFFFFFFFFFFFFFFF
print('65-bit value:', val)
print('Bit length:', val.bit_length())
print('Type:', type(val))
print('Hex:', hex(val))

# Test the actual operation from Hypothesis
sign = 1
float_lex = 0xFFFFFFFFFFFFFFFF
result = (sign << 64) | float_lex
print('\nHypothesis operation:')
print('Result:', result)
print('Result bit length:', result.bit_length())
print('Result hex:', hex(result))

# Test extracting back
sign_back = result >> 64
lex_back = result & ((1 << 64) - 1)
print('\nExtracting back:')
print('Sign back:', sign_back)
print('Lex back:', hex(lex_back))
print('Round trip works:', sign == sign_back and float_lex == lex_back)

# Test with different sign values  
print('\nTesting with sign=0:')
sign0_result = (0 << 64) | float_lex
print('Sign=0 result:', hex(sign0_result))
print('Sign=0 bit length:', sign0_result.bit_length())

# Test maximum possible value
print('\nTesting maximum 65-bit value:')
max_65_bit = (1 << 65) - 1
print('Max 65-bit value:', hex(max_65_bit))
print('Max 65-bit bit length:', max_65_bit.bit_length())

# Test sys.maxsize to see native int limits
import sys
print('\nSystem info:')
print('sys.maxsize:', sys.maxsize)
print('sys.maxsize bit length:', sys.maxsize.bit_length())
print('Python supports arbitrary precision integers: True')