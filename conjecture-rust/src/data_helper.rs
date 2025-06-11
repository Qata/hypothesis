use crate::choice::{IntervalSet};

/// Convert IntervalSet to alphabet string (standalone function to avoid borrowing issues)
pub fn intervals_to_alphabet_static(intervals: &IntervalSet) -> String {
    let mut alphabet = String::new();
    for &(start, end) in &intervals.intervals {
        for codepoint in start..=end.min(0x10FFFF) { // Limit to valid Unicode
            if let Some(ch) = char::from_u32(codepoint) {
                if is_valid_string_char_static(ch) {
                    alphabet.push(ch);
                }
            }
            // Limit alphabet size to prevent memory issues
            if alphabet.len() > 10000 {
                break;
            }
        }
        if alphabet.len() > 10000 {
            break;
        }
    }
    alphabet
}

/// Static version of is_valid_string_char
fn is_valid_string_char_static(ch: char) -> bool {
    let codepoint = ch as u32;
    
    // Filter out problematic Unicode categories:
    match codepoint {
        // Control characters (except tab, LF, CR)
        0x00..=0x08 => false,
        0x0B..=0x0C => false,
        0x0E..=0x1F => false,
        0x7F..=0x9F => false,
        // Surrogates
        0xD800..=0xDFFF => false,
        // Private use areas
        0xE000..=0xF8FF => false,
        0xF0000..=0xFFFFD => false,
        0x100000..=0x10FFFD => false,
        _ => true,
    }
}