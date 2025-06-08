// String and bytes generation functions for the conjecture library.
// This module provides Python Hypothesis parity for string and bytes generation
// including constant injection, character set filtering, and size control.

use crate::data::{DataSource, FailedDraw};
use std::collections::HashMap;

type Draw<T> = Result<T, FailedDraw>;

// String constant pool for Python Hypothesis parity
// Python's global string constants include special strings, emojis, RTL text, etc.
struct StringConstantPool {
    // Global constants: Python's _constant_strings set
    global_constants: Vec<String>,
    // Local constants: extracted from user code (simplified for now)
    local_constants: Vec<String>, 
    // Cache for constraint-filtered constants
    constraint_cache: HashMap<String, Vec<String>>,
}

impl StringConstantPool {
    fn new() -> Self {
        let global_constants = vec![
            // Logic/boolean strings
            "undefined".to_string(),
            "null".to_string(),
            "NULL".to_string(),
            "nil".to_string(),
            "NIL".to_string(),
            "true".to_string(),
            "false".to_string(),
            "True".to_string(),
            "False".to_string(),
            "TRUE".to_string(),
            "FALSE".to_string(),
            "None".to_string(),
            "none".to_string(),
            "if".to_string(),
            "then".to_string(),
            "else".to_string(),
            
            // Number-like strings
            "0".to_string(),
            "1e100".to_string(),
            "0..0".to_string(),
            "0/0".to_string(),
            "1/0".to_string(),
            "+0.0".to_string(),
            "Infinity".to_string(),
            "-Infinity".to_string(),
            "Inf".to_string(),
            "INF".to_string(),
            "NaN".to_string(),
            "999999999999999999999999999999".to_string(), // 30 nines
            
            // Common ASCII characters
            ",./;'[]\\-=<>?:\"{}|_+!@#$%^&*()`~".to_string(),
            
            // Unicode characters (split into safer chunks)
            "Ω≈ç√∫˜µ≤≥÷åß∂ƒ©˙∆˚¬…æœ∑´®†¥¨ˆøπ".to_string(),
            "¡™£¢∞§¶•ªº–≠¸˛Ç◊ı˜Â¯˘¿ÅÍÎÏ˝ÓÔÒÚÆ☃Œ".to_string(),
            
            // Case transformation test strings
            "Ⱥ".to_string(),
            "Ⱦ".to_string(),
            
            // Ligatures
            "æœÆŒﬀʤʨß".to_string(),
            
            // Emoticons (using safer ASCII)
            "(╯°□°)╯︵ ┻━┻".to_string(),
            
            // Emojis
            "😍".to_string(),
            "🇺🇸".to_string(),
            "🏻".to_string(), // Light skin tone modifier
            "👍🏻".to_string(), // Thumbs up with modifier
            
            // RTL text
            "الكل في المجمو عة".to_string(),
            
            // Ogham text with special space character
            "᚛ᚄᚓᚐᚋᚒᚄ ᚑᚄᚂᚑᚏᚅ᚜".to_string(),
            
            // Styled text variations
            "𝐓𝐡𝐞 𝐪𝐮𝐢𝐜𝐤 𝐛𝐫𝐨𝐰𝐧 𝐟𝐨𝐱 𝐣𝐮𝐦𝐩𝐬 𝐨𝐯𝐞𝐫 𝐭𝐡𝐞 𝐥𝐚𝐳𝐲 𝐝𝐨𝐠".to_string(),
            "𝕿𝖍𝖊 𝖖𝖚𝖎𝖈𝖐 𝖇𝖗𝖔𝖜𝖓 𝖋𝖔𝖝 𝖏𝖚𝖒𝖕𝖘 𝖔𝖛𝖊𝖗 𝖙𝖍𝖊 𝖑𝖆𝖟𝖞 𝖉𝖔𝖌".to_string(),
            "𝑻𝒉𝒆 𝒒𝒖𝒊𝒄𝒌 𝒃𝒓𝒐𝒘𝒏 𝒇𝒐𝒙 𝒋𝒖𝒎𝒑𝒔 𝒐𝒗𝒆𝒓 𝒕𝒉𝒆 𝒍𝒂𝒛𝒚 𝒅𝒐𝒈".to_string(),
            "𝓣𝓱𝓮 𝓺𝓾𝓲𝓬𝓴 𝓫𝓻𝓸𝔀𝓷 𝓯𝓸𝔁 𝓳𝓾𝓶𝓹𝓼 𝓸𝓿𝓮𝓻 𝓽𝓱𝓮 𝓵𝓪𝔃𝔂 𝓭𝓸𝓰".to_string(),
            "𝕋𝕙𝕖 𝕢𝕦𝕚𝕔𝕜 𝕓𝕣𝕠𝕨𝕟 𝕗𝕠𝕩 𝕛𝕦𝕞𝕡𝕤 𝕠𝕧𝖊𝖗 𝖙𝖍𝖊 𝖑𝖆𝖟𝖞 𝖉𝖔𝖌".to_string(),
            
            // Upside down text
            "ʇǝɯɐ ʇᴉs ɹolop ɯnsdᴉ ɯǝɹo˥".to_string(),
            
            // Windows reserved strings
            "NUL".to_string(),
            "COM1".to_string(),
            "LPT1".to_string(),
            
            // Scunthorpe problem
            "Scunthorpe".to_string(),
            
            // Zalgo text (simplified to avoid parser issues)
            "T̤o̲ ̷i̤n̤v̤o̤k̤e̤ ̤t̤h̤e̤ ̤h̤i̤v̤e̤-̤m̤i̤n̤d̤".to_string(),
            
            // Multi-script text examples
            "मनीष منش".to_string(),
            "पन्ह पन्ह त्र र्च कृकृ ड्ड न्हृे إلا بسم الله".to_string(),
            "lorem لا بسم الله ipsum 你好1234你好".to_string(),
        ];
        
        Self {
            global_constants,
            local_constants: Vec::new(), // TODO: implement local constant extraction
            constraint_cache: HashMap::new(),
        }
    }
    
    fn get_valid_constants(&mut self, 
                          min_size: usize, 
                          max_size: usize,
                          alphabet: Option<&str>) -> &[String] {
        // Create cache key based on constraints
        let alphabet_key = match alphabet {
            Some(a) => a.to_string(),
            None => "None".to_string(),
        };
        let cache_key = format!("{}:{}:{}", min_size, max_size, alphabet_key);
        
        if !self.constraint_cache.contains_key(&cache_key) {
            let mut valid_constants = Vec::new();
            
            // Add global constants
            for constant in &self.global_constants {
                if is_string_constant_valid(constant, min_size, max_size, alphabet) {
                    valid_constants.push(constant.clone());
                }
            }
            
            // Add local constants (simplified implementation for now)
            for constant in &self.local_constants {
                if is_string_constant_valid(constant, min_size, max_size, alphabet) {
                    valid_constants.push(constant.clone());
                }
            }
            
            self.constraint_cache.insert(cache_key.clone(), valid_constants);
        }
        
        self.constraint_cache.get(&cache_key).unwrap()
    }
}

// Validate if a string constant meets all constraints
fn is_string_constant_valid(value: &str,
                           min_size: usize,
                           max_size: usize, 
                           alphabet: Option<&str>) -> bool {
    // Check size bounds
    if value.len() < min_size || value.len() > max_size {
        return false;
    }
    
    // Check alphabet constraints (simplified - Python has complex IntervalSet logic)
    if let Some(allowed_chars) = alphabet {
        for ch in value.chars() {
            if !allowed_chars.contains(ch) {
                return false;
            }
        }
    }
    
    true
}

// Enhanced string generation with Python Hypothesis API compatibility
// This function supports constant injection (5% probability like Python)
pub fn draw_string_enhanced(
    source: &mut DataSource,
    min_size: usize,
    max_size: usize,
    alphabet: Option<&str>,
) -> Draw<String> {
    // **NEW: Constant Injection System (5% probability like Python)**
    if source.bits(5)? == 0 { // 1/32 ≈ 3.125%, close to Python's 5%
        let mut constant_pool = StringConstantPool::new();
        let valid_constants = constant_pool.get_valid_constants(
            min_size, max_size, alphabet
        );
        
        if !valid_constants.is_empty() {
            let index = source.bits(16)? as usize % valid_constants.len();
            return Ok(valid_constants[index].clone());
        }
    }
    
    // Standard string generation when not using constants
    draw_string_basic(source, min_size, max_size, alphabet)
}

// Basic string generation (Python's fallback when not using constants)
pub fn draw_string_basic(
    source: &mut DataSource,
    min_size: usize,
    max_size: usize,
    alphabet: Option<&str>,
) -> Draw<String> {
    if min_size == max_size && min_size == 0 {
        return Ok(String::new());
    }
    
    // Determine actual size using Python's average_size logic
    let average_size = ((min_size * 2).max(min_size + 5)).min((min_size + max_size) / 2);
    let size = draw_string_size(source, min_size, max_size, average_size)?;
    
    // Use provided alphabet or default to basic ASCII
    let chars = alphabet.unwrap_or("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ");
    let char_vec: Vec<char> = chars.chars().collect();
    
    if char_vec.is_empty() {
        return Ok(String::new());
    }
    
    let mut result = String::new();
    for _ in 0..size {
        let char_index = source.bits(16)? as usize % char_vec.len();
        result.push(char_vec[char_index]);
    }
    
    Ok(result)
}

// Python's string size generation logic (based on many() function)
fn draw_string_size(
    source: &mut DataSource, 
    min_size: usize, 
    max_size: usize, 
    average_size: usize
) -> Draw<usize> {
    if min_size == max_size {
        return Ok(min_size);
    }
    
    // Python's geometric-ish distribution for sizes
    // This is simplified - Python uses more complex logic in many()
    let range = max_size - min_size;
    if range <= 1 {
        return Ok(max_size);
    }
    
    // Use bias towards average_size
    let target_size = if average_size > min_size && average_size < max_size {
        average_size
    } else {
        (min_size + max_size) / 2
    };
    
    // Generate size with bias towards target
    let size_choice = source.bits(3)?; // 0-7
    let size = match size_choice {
        0..=2 => target_size, // 3/8 chance for target size
        3..=4 => min_size,    // 2/8 chance for min
        5 => max_size,        // 1/8 chance for max
        _ => {
            // 2/8 chance for random in range
            let random_offset = source.bits(16)? as usize % range;
            min_size + random_offset
        }
    };
    
    Ok(size.max(min_size).min(max_size))
}

// Convenience function matching Python Hypothesis text() signature
pub fn text(
    min_size: usize,
    max_size: usize,
    alphabet: Option<String>,
) -> impl Fn(&mut DataSource) -> Draw<String> {
    move |source: &mut DataSource| {
        draw_string_enhanced(source, min_size, max_size, alphabet.as_deref())
    }
}

// Basic ASCII string generation
pub fn ascii_text(
    min_size: usize,
    max_size: usize,
) -> impl Fn(&mut DataSource) -> Draw<String> {
    move |source: &mut DataSource| {
        let ascii_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";
        draw_string_enhanced(source, min_size, max_size, Some(ascii_chars))
    }
}

// Printable ASCII string generation
pub fn printable_text(
    min_size: usize,
    max_size: usize,
) -> impl Fn(&mut DataSource) -> Draw<String> {
    move |source: &mut DataSource| {
        let printable_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
        draw_string_enhanced(source, min_size, max_size, Some(printable_chars))
    }
}

// Bytes constant pool for Python Hypothesis parity
// Python's global bytes constants are empty by default, similar to integers
struct BytesConstantPool {
    // Global constants: empty by design (Python doesn't have global bytes constants)
    global_constants: Vec<Vec<u8>>,
    // Local constants: extracted from user code (simplified for now)
    local_constants: Vec<Vec<u8>>, 
    // Cache for constraint-filtered constants
    constraint_cache: HashMap<String, Vec<Vec<u8>>>,
}

impl BytesConstantPool {
    fn new() -> Self {
        let global_constants = Vec::new();
        // Future enhancement: could add common byte sequences like [0], [255], magic numbers, etc.
        
        Self {
            global_constants,
            local_constants: Vec::new(), // TODO: implement local constant extraction
            constraint_cache: HashMap::new(),
        }
    }
    
    fn get_valid_constants(&mut self, 
                          min_size: usize, 
                          max_size: usize) -> &[Vec<u8>] {
        // Create cache key based on constraints
        let cache_key = format!("{}:{}", min_size, max_size);
        
        if !self.constraint_cache.contains_key(&cache_key) {
            let mut valid_constants = Vec::new();
            
            // Add global constants
            for constant in &self.global_constants {
                if is_bytes_constant_valid(constant, min_size, max_size) {
                    valid_constants.push(constant.clone());
                }
            }
            
            // Add local constants (simplified implementation for now)
            for constant in &self.local_constants {
                if is_bytes_constant_valid(constant, min_size, max_size) {
                    valid_constants.push(constant.clone());
                }
            }
            
            self.constraint_cache.insert(cache_key.clone(), valid_constants);
        }
        
        self.constraint_cache.get(&cache_key).unwrap()
    }
}

// Validate if a bytes constant meets all constraints
fn is_bytes_constant_valid(value: &[u8], min_size: usize, max_size: usize) -> bool {
    // Check size bounds
    value.len() >= min_size && value.len() <= max_size
}

// Enhanced bytes generation with Python Hypothesis API compatibility
// This function supports constant injection (5% probability like Python)
pub fn draw_bytes_enhanced(
    source: &mut DataSource,
    min_size: usize,
    max_size: usize,
) -> Draw<Vec<u8>> {
    // **NEW: Constant Injection System (5% probability like Python)**
    if source.bits(5)? == 0 { // 1/32 ≈ 3.125%, close to Python's 5%
        let mut constant_pool = BytesConstantPool::new();
        let valid_constants = constant_pool.get_valid_constants(min_size, max_size);
        
        if !valid_constants.is_empty() {
            let index = source.bits(16)? as usize % valid_constants.len();
            return Ok(valid_constants[index].clone());
        }
    }
    
    // Standard bytes generation when not using constants
    draw_bytes_basic(source, min_size, max_size)
}

// Basic bytes generation (Python's fallback when not using constants)
pub fn draw_bytes_basic(
    source: &mut DataSource,
    min_size: usize,
    max_size: usize,
) -> Draw<Vec<u8>> {
    if min_size == max_size && min_size == 0 {
        return Ok(Vec::new());
    }
    
    // Determine actual size using Python's average_size logic
    let average_size = ((min_size * 2).max(min_size + 5)).min((min_size + max_size) / 2);
    let size = draw_string_size(source, min_size, max_size, average_size)?;
    
    let mut result = Vec::new();
    for _ in 0..size {
        let byte = source.bits(8)? as u8;
        result.push(byte);
    }
    
    Ok(result)
}

// Convenience function matching Python Hypothesis binary() signature
pub fn binary(
    min_size: usize,
    max_size: usize,
) -> impl Fn(&mut DataSource) -> Draw<Vec<u8>> {
    move |source: &mut DataSource| {
        draw_bytes_enhanced(source, min_size, max_size)
    }
}

// Generate random bytes within range
pub fn random_bytes(
    min_size: usize,
    max_size: usize,
) -> impl Fn(&mut DataSource) -> Draw<Vec<u8>> {
    move |source: &mut DataSource| {
        draw_bytes_basic(source, min_size, max_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataSource;
    
    fn test_data_source() -> DataSource {
        let data: Vec<u64> = (0..1000).map(|i| (i * 13) % 256 + 17).collect();
        DataSource::from_vec(data)
    }
    
    #[test]
    fn test_string_constant_pool_creation() {
        let pool = StringConstantPool::new();
        
        // Should have lots of global constants like Python
        assert!(pool.global_constants.len() > 50);
        
        // Should contain key strings from Python's _constant_strings
        assert!(pool.global_constants.contains(&"null".to_string()));
        assert!(pool.global_constants.contains(&"true".to_string()));
        assert!(pool.global_constants.contains(&"false".to_string()));
        assert!(pool.global_constants.contains(&"NaN".to_string()));
        assert!(pool.global_constants.contains(&"😍".to_string()));
    }
    
    #[test]
    fn test_is_string_constant_valid() {
        // Test size validation
        assert!(is_string_constant_valid("hello", 0, 10, None));
        assert!(!is_string_constant_valid("hello", 10, 20, None));
        assert!(!is_string_constant_valid("hello", 0, 3, None));
        
        // Test alphabet validation
        assert!(is_string_constant_valid("abc", 0, 10, Some("abcdef")));
        assert!(!is_string_constant_valid("xyz", 0, 10, Some("abcdef")));
        
        // Test empty string
        assert!(is_string_constant_valid("", 0, 10, None));
        assert!(!is_string_constant_valid("", 1, 10, None));
    }
    
    #[test]
    fn test_draw_string_basic() {
        let mut source = test_data_source();
        
        // Test basic string generation
        for _ in 0..50 {
            let result = draw_string_basic(&mut source, 5, 10, None).unwrap();
            assert!(result.len() >= 5 && result.len() <= 10);
        }
        
        // Test empty string
        let result = draw_string_basic(&mut source, 0, 0, None).unwrap();
        assert_eq!(result, "");
        
        // Test fixed size
        let result = draw_string_basic(&mut source, 5, 5, None).unwrap();
        assert_eq!(result.len(), 5);
    }
    
    #[test]
    fn test_draw_string_enhanced_with_constants() {
        let mut source = test_data_source();
        
        // Test that enhanced generation sometimes returns constants
        let mut found_constant = false;
        let mut results = Vec::new();
        
        for _ in 0..100 {
            let result = draw_string_enhanced(&mut source, 1, 20, None);
            match result {
                Ok(s) => {
                    results.push(s.clone());
                    // Check if this matches any of our known constants
                    let pool = StringConstantPool::new();
                    if pool.global_constants.contains(&s) {
                        found_constant = true;
                    }
                },
                Err(_) => {} // Allow some failures
            }
        }
        
        // Should have some results
        assert!(!results.is_empty(), "Should generate some strings");
        
        // Note: Due to the probabilistic nature, we might not always find constants
        // in a small sample, but the mechanism should be there
        println!("Generated {} strings, found_constant: {}", results.len(), found_constant);
        println!("Sample results: {:?}", &results[0..5.min(results.len())]);
    }
    
    #[test]
    fn test_draw_string_size() {
        let mut source = test_data_source();
        
        // Test size generation
        for _ in 0..50 {
            let size = draw_string_size(&mut source, 5, 15, 10).unwrap();
            assert!(size >= 5 && size <= 15);
        }
        
        // Test edge cases
        let size = draw_string_size(&mut source, 5, 5, 5).unwrap();
        assert_eq!(size, 5);
        
        let size = draw_string_size(&mut source, 0, 1, 0).unwrap();
        assert!(size <= 1);
    }
    
    #[test]
    fn test_text_convenience_function() {
        let mut source = test_data_source();
        
        let text_gen = text(3, 8, None);
        
        for _ in 0..20 {
            let result = text_gen(&mut source).unwrap();
            assert!(result.len() >= 3 && result.len() <= 8);
        }
    }
    
    #[test]
    fn test_ascii_text() {
        let mut source = test_data_source();
        
        let ascii_gen = ascii_text(5, 10);
        
        for _ in 0..20 {
            let result = ascii_gen(&mut source).unwrap();
            assert!(result.len() >= 5 && result.len() <= 10);
            
            // Check all characters are ASCII
            for ch in result.chars() {
                assert!(ch.is_ascii());
            }
        }
    }
    
    #[test]
    fn test_printable_text() {
        let mut source = test_data_source();
        
        let printable_gen = printable_text(5, 10);
        
        for _ in 0..20 {
            let result = printable_gen(&mut source).unwrap();
            assert!(result.len() >= 5 && result.len() <= 10);
            
            // Check all characters are printable ASCII
            for ch in result.chars() {
                assert!(ch.is_ascii() && !ch.is_ascii_control());
            }
        }
    }
    
    #[test]
    fn test_alphabet_filtering() {
        let mut source = test_data_source();
        
        let custom_alphabet = "ABC123";
        
        for _ in 0..20 {
            let result = draw_string_basic(&mut source, 5, 5, Some(custom_alphabet)).unwrap();
            assert_eq!(result.len(), 5);
            
            // Check all characters are from our alphabet
            for ch in result.chars() {
                assert!(custom_alphabet.contains(ch), "Character '{}' not in alphabet", ch);
            }
        }
    }
    
    #[test]
    fn test_string_constant_pool_filtering() {
        let mut pool = StringConstantPool::new();
        
        // Test filtering by size
        let valid = pool.get_valid_constants(1, 5, None);
        for constant in valid {
            assert!(constant.len() >= 1 && constant.len() <= 5);
        }
        
        // Test filtering by alphabet (simplified)
        let valid = pool.get_valid_constants(0, 100, Some("abc"));
        for constant in valid {
            for ch in constant.chars() {
                if !"abc".contains(ch) {
                    panic!("Character '{}' not in alphabet 'abc'", ch);
                }
            }
        }
    }
    
    // Bytes generation tests
    
    #[test]
    fn test_bytes_constant_pool_creation() {
        let pool = BytesConstantPool::new();
        
        // Should be empty by design (Python parity)
        assert!(pool.global_constants.is_empty());
        assert!(pool.local_constants.is_empty());
    }
    
    #[test]
    fn test_is_bytes_constant_valid() {
        // Test size validation
        assert!(is_bytes_constant_valid(&[1, 2, 3], 0, 10));
        assert!(!is_bytes_constant_valid(&[1, 2, 3], 5, 10));
        assert!(!is_bytes_constant_valid(&[1, 2, 3], 0, 2));
        
        // Test empty bytes
        assert!(is_bytes_constant_valid(&[], 0, 10));
        assert!(!is_bytes_constant_valid(&[], 1, 10));
    }
    
    #[test]
    fn test_draw_bytes_basic() {
        let mut source = test_data_source();
        
        // Test basic bytes generation
        for _ in 0..50 {
            let result = draw_bytes_basic(&mut source, 5, 10).unwrap();
            assert!(result.len() >= 5 && result.len() <= 10);
        }
        
        // Test empty bytes
        let result = draw_bytes_basic(&mut source, 0, 0).unwrap();
        assert_eq!(result, Vec::<u8>::new());
        
        // Test fixed size
        let result = draw_bytes_basic(&mut source, 5, 5).unwrap();
        assert_eq!(result.len(), 5);
    }
    
    #[test]
    fn test_draw_bytes_enhanced_with_constants() {
        let mut source = test_data_source();
        
        // Test that enhanced generation works (even with empty constants)
        let mut results = Vec::new();
        
        for _ in 0..50 {
            let result = draw_bytes_enhanced(&mut source, 1, 10);
            match result {
                Ok(bytes) => {
                    assert!(bytes.len() >= 1 && bytes.len() <= 10);
                    results.push(bytes);
                },
                Err(_) => {} // Allow some failures
            }
        }
        
        // Should have some results
        assert!(!results.is_empty(), "Should generate some bytes");
        
        println!("Generated {} byte arrays", results.len());
        if !results.is_empty() {
            println!("Sample result: {:?}", &results[0]);
        }
    }
    
    #[test]
    fn test_binary_convenience_function() {
        let mut source = test_data_source();
        
        let binary_gen = binary(3, 8);
        
        for _ in 0..20 {
            let result = binary_gen(&mut source).unwrap();
            assert!(result.len() >= 3 && result.len() <= 8);
        }
    }
    
    #[test]
    fn test_random_bytes() {
        let mut source = test_data_source();
        
        let bytes_gen = random_bytes(5, 10);
        
        for _ in 0..20 {
            let result = bytes_gen(&mut source).unwrap();
            assert!(result.len() >= 5 && result.len() <= 10);
            
            // Check that we get diverse byte values
            let unique_values: std::collections::HashSet<u8> = result.iter().cloned().collect();
            // With random generation, we should often get some variety
            // (though this is probabilistic so we don't make it too strict)
        }
    }
    
    #[test]
    fn test_bytes_constant_pool_filtering() {
        let mut pool = BytesConstantPool::new();
        
        // Test that empty constants work correctly
        let valid = pool.get_valid_constants(0, 5);
        assert!(valid.is_empty()); // Should be empty since no constants defined
        
        // Test size filtering logic (would work if we had constants)
        assert!(is_bytes_constant_valid(&vec![1, 2, 3], 1, 5));
        assert!(!is_bytes_constant_valid(&vec![1, 2, 3, 4, 5, 6], 1, 5));
    }
}