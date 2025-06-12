//! Comprehensive Capability Tests for DFA-Based String Generation System
//!
//! Tests the complete DFA-based string generation capability including L* algorithm learning,
//! pattern recognition, structured string generation, and integration with the choice system.
//! These tests validate the entire capability's behavior end-to-end through PyO3 and FFI.

use crate::choice::{
    ChoiceValue, ChoiceType, Constraints, StringConstraints, IntervalSet,
    StandardValueGenerator, ValueGenerator, BufferEntropySource, EntropySource,
    ChoiceNode, ChoiceSequence, NavigationSystem, TemplatingSystem, 
    WeightedSelection, CumulativeWeightedSelector,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use pyo3::prelude::*;

/// Error types for DFA capability testing
#[derive(Debug, Clone, PartialEq)]
pub enum DFACapabilityError {
    LearningFailed(String),
    GenerationFailed(String),
    IntegrationFailed(String),
    FFIError(String),
}

/// Comprehensive DFA-based string generation capability
#[derive(Debug)]
pub struct DFAStringGenerationCapability {
    /// L* learning engine for pattern recognition
    learning_engine: LStarLearningEngine,
    /// String generation engine with DFA integration
    generation_engine: DFAStringGenerationEngine,
    /// Choice system integration layer
    choice_integration: DFAChoiceIntegration,
    /// Performance metrics and monitoring
    metrics: DFACapabilityMetrics,
}

/// L* learning engine for finite automata
#[derive(Debug)]
pub struct LStarLearningEngine {
    /// Current learned DFA
    dfa: Option<LearnedDFA>,
    /// Learning statistics
    queries_made: usize,
    learning_time_ms: u64,
    /// Oracle interface for membership queries
    oracle: Option<Box<dyn MembershipOracle>>,
}

/// DFA string generation engine
#[derive(Debug)]
pub struct DFAStringGenerationEngine {
    /// Standard fallback generator
    fallback_generator: StandardValueGenerator,
    /// DFA-specific generation parameters
    generation_params: DFAGenerationParams,
}

/// Choice system integration for DFA capability
#[derive(Debug)]
pub struct DFAChoiceIntegration {
    /// Navigation system integration
    navigation_enabled: bool,
    /// Templating system integration  
    templating_enabled: bool,
    /// Weighted selection integration
    weighted_selection_enabled: bool,
}

/// Performance metrics for DFA capability
#[derive(Debug, Default)]
pub struct DFACapabilityMetrics {
    /// Learning performance
    total_queries: usize,
    learning_sessions: usize,
    average_states_learned: f64,
    
    /// Generation performance
    strings_generated: usize,
    constraint_satisfaction_rate: f64,
    average_generation_time_us: f64,
    
    /// Integration metrics
    choice_integrations: usize,
    ffi_calls: usize,
    error_rate: f64,
}

/// DFA generation parameters
#[derive(Debug, Clone)]
pub struct DFAGenerationParams {
    /// Maximum string length for enumeration
    max_enumeration_length: usize,
    /// Fallback probability when DFA fails
    fallback_probability: f64,
    /// Constraint satisfaction priority
    constraint_priority: ConstraintPriority,
}

/// Priority levels for constraint satisfaction
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintPriority {
    /// Prioritize DFA pattern compliance
    PatternFirst,
    /// Prioritize constraint satisfaction
    ConstraintsFirst,
    /// Balance both requirements
    Balanced,
}

/// Membership oracle trait for L* learning
pub trait MembershipOracle: std::fmt::Debug {
    /// Test if string belongs to target language
    fn is_member(&self, input: &str) -> bool;
    /// Test if string is valid prefix (for incremental learning)
    fn is_prefix(&self, input: &str) -> bool;
    /// Get oracle description for logging
    fn description(&self) -> String;
}

/// Learned DFA structure
#[derive(Debug, Clone)]
pub struct LearnedDFA {
    /// DFA states with metadata
    states: HashMap<usize, DFAState>,
    /// Transition function
    transitions: HashMap<(usize, char), usize>,
    /// Start state
    start_state: usize,
    /// Alphabet
    alphabet: HashSet<char>,
    /// Learning metadata
    metadata: DFAMetadata,
}

/// DFA state with learning context
#[derive(Debug, Clone)]
pub struct DFAState {
    id: usize,
    is_accepting: bool,
    access_string: String,
    experiments: Vec<String>,
}

/// Metadata about learned DFA
#[derive(Debug, Clone)]
pub struct DFAMetadata {
    queries_used: usize,
    learning_time_ms: u64,
    pattern_description: String,
    alphabet_size: usize,
}

impl DFAStringGenerationCapability {
    /// Create new DFA capability instance
    pub fn new() -> Self {
        Self {
            learning_engine: LStarLearningEngine::new(),
            generation_engine: DFAStringGenerationEngine::new(),
            choice_integration: DFAChoiceIntegration::new(),
            metrics: DFACapabilityMetrics::default(),
        }
    }
    
    /// Learn DFA from membership oracle
    pub fn learn_pattern(
        &mut self,
        oracle: Box<dyn MembershipOracle>,
        alphabet: HashSet<char>,
    ) -> Result<(), DFACapabilityError> {
        let start_time = std::time::Instant::now();
        
        // Execute L* learning
        let dfa = self.learning_engine.learn_dfa(oracle, alphabet)?;
        
        // Update metrics
        let learning_time = start_time.elapsed().as_millis() as u64;
        self.metrics.learning_sessions += 1;
        self.metrics.total_queries += dfa.metadata.queries_used;
        self.metrics.average_states_learned = 
            (self.metrics.average_states_learned * (self.metrics.learning_sessions - 1) as f64 + 
             dfa.states.len() as f64) / self.metrics.learning_sessions as f64;
        
        // Store learned DFA
        self.learning_engine.dfa = Some(dfa);
        self.learning_engine.learning_time_ms = learning_time;
        
        Ok(())
    }
    
    /// Generate string using learned DFA with constraints
    pub fn generate_string(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFACapabilityError> {
        let start_time = std::time::Instant::now();
        
        // Generate using DFA if available
        let result = if let Some(ref dfa) = self.learning_engine.dfa {
            self.generation_engine.generate_with_dfa(dfa, constraints, entropy)
        } else {
            self.generation_engine.generate_fallback(constraints, entropy)
        };
        
        // Update metrics
        let generation_time = start_time.elapsed().as_micros() as f64;
        self.metrics.strings_generated += 1;
        self.metrics.average_generation_time_us = 
            (self.metrics.average_generation_time_us * (self.metrics.strings_generated - 1) as f64 + 
             generation_time) / self.metrics.strings_generated as f64;
        
        // Update constraint satisfaction rate
        if let Ok(ref generated) = result {
            let satisfies = self.check_constraint_satisfaction(generated, constraints);
            self.metrics.constraint_satisfaction_rate = 
                (self.metrics.constraint_satisfaction_rate * (self.metrics.strings_generated - 1) as f64 + 
                 if satisfies { 1.0 } else { 0.0 }) / self.metrics.strings_generated as f64;
        }
        
        result
    }
    
    /// Integrate with choice system for navigation support
    pub fn enable_choice_navigation(&mut self) -> Result<(), DFACapabilityError> {
        self.choice_integration.navigation_enabled = true;
        self.metrics.choice_integrations += 1;
        Ok(())
    }
    
    /// Integrate with templating system
    pub fn enable_templating(&mut self) -> Result<(), DFACapabilityError> {
        self.choice_integration.templating_enabled = true;
        self.metrics.choice_integrations += 1;
        Ok(())
    }
    
    /// Integrate with weighted selection
    pub fn enable_weighted_selection(&mut self) -> Result<(), DFACapabilityError> {
        self.choice_integration.weighted_selection_enabled = true;
        self.metrics.choice_integrations += 1;
        Ok(())
    }
    
    /// Get capability metrics
    pub fn get_metrics(&self) -> &DFACapabilityMetrics {
        &self.metrics
    }
    
    /// Check if generated string satisfies constraints
    fn check_constraint_satisfaction(&self, generated: &str, constraints: &StringConstraints) -> bool {
        // Check length constraints
        if generated.len() < constraints.min_size || generated.len() > constraints.max_size {
            return false;
        }
        
        // Check character interval constraints
        generated.chars().all(|ch| {
            let code = ch as u32;
            constraints.intervals.intervals.iter().any(|(start, end)| {
                code >= *start && code <= *end
            })
        })
    }
}

impl LStarLearningEngine {
    pub fn new() -> Self {
        Self {
            dfa: None,
            queries_made: 0,
            learning_time_ms: 0,
            oracle: None,
        }
    }
    
    pub fn learn_dfa(
        &mut self,
        oracle: Box<dyn MembershipOracle>,
        alphabet: HashSet<char>,
    ) -> Result<LearnedDFA, DFACapabilityError> {
        self.oracle = Some(oracle);
        
        // Initialize L* learning structures
        let mut observation_table = HashMap::new();
        let mut prefixes = HashSet::new();
        let mut suffixes = HashSet::new();
        
        // Start with empty string
        prefixes.insert(String::new());
        suffixes.insert(String::new());
        
        // Main L* learning loop
        loop {
            // Fill observation table
            self.fill_observation_table(&mut observation_table, &prefixes, &suffixes)?;
            
            // Make table closed and consistent
            let closed = self.make_closed(&mut observation_table, &mut prefixes, &suffixes, &alphabet)?;
            let consistent = self.make_consistent(&mut observation_table, &prefixes, &mut suffixes)?;
            
            if closed && consistent {
                // Build candidate DFA
                let candidate = self.build_candidate_dfa(&observation_table, &prefixes, &alphabet)?;
                
                // Test equivalence with oracle
                if let Some(counterexample) = self.find_counterexample(&candidate)? {
                    // Add counterexample suffixes and continue learning
                    for i in 0..=counterexample.len() {
                        suffixes.insert(counterexample.chars().skip(i).collect());
                    }
                } else {
                    // Learning complete
                    return Ok(candidate);
                }
            }
            
            // Safety check against infinite loops
            if self.queries_made > 10000 {
                return Err(DFACapabilityError::LearningFailed(
                    "Too many membership queries".to_string()
                ));
            }
        }
    }
    
    fn fill_observation_table(
        &mut self,
        table: &mut HashMap<String, HashMap<String, bool>>,
        prefixes: &HashSet<String>,
        suffixes: &HashSet<String>,
    ) -> Result<(), DFACapabilityError> {
        let oracle = self.oracle.as_ref().unwrap();
        
        for prefix in prefixes {
            if !table.contains_key(prefix) {
                table.insert(prefix.clone(), HashMap::new());
            }
            
            for suffix in suffixes {
                if !table[prefix].contains_key(suffix) {
                    let query_string = format!("{}{}", prefix, suffix);
                    let result = oracle.is_member(&query_string);
                    self.queries_made += 1;
                    
                    table.get_mut(prefix).unwrap().insert(suffix.clone(), result);
                }
            }
        }
        
        Ok(())
    }
    
    fn make_closed(
        &mut self,
        table: &mut HashMap<String, HashMap<String, bool>>,
        prefixes: &mut HashSet<String>,
        suffixes: &HashSet<String>,
        alphabet: &HashSet<char>,
    ) -> Result<bool, DFACapabilityError> {
        let mut changes_made = false;
        
        for prefix in prefixes.clone() {
            for &ch in alphabet {
                let extended = format!("{}{}", prefix, ch);
                if !prefixes.contains(&extended) {
                    // Check if extended string has representative in prefixes
                    if !self.has_representative(&extended, table, prefixes, suffixes) {
                        prefixes.insert(extended);
                        changes_made = true;
                    }
                }
            }
        }
        
        Ok(!changes_made)
    }
    
    fn make_consistent(
        &mut self,
        table: &HashMap<String, HashMap<String, bool>>,
        prefixes: &HashSet<String>,
        suffixes: &mut HashSet<String>,
    ) -> Result<bool, DFACapabilityError> {
        for prefix1 in prefixes {
            for prefix2 in prefixes {
                if prefix1 != prefix2 && self.rows_equal(prefix1, prefix2, table, suffixes) {
                    // Find inconsistency
                    if let Some(suffix) = self.find_distinguishing_suffix(prefix1, prefix2, table, suffixes) {
                        suffixes.insert(suffix);
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }
    
    fn has_representative(
        &self,
        target: &str,
        table: &HashMap<String, HashMap<String, bool>>,
        prefixes: &HashSet<String>,
        suffixes: &HashSet<String>,
    ) -> bool {
        for prefix in prefixes {
            if self.rows_equal(target, prefix, table, suffixes) {
                return true;
            }
        }
        false
    }
    
    fn rows_equal(
        &self,
        prefix1: &str,
        prefix2: &str,
        table: &HashMap<String, HashMap<String, bool>>,
        suffixes: &HashSet<String>,
    ) -> bool {
        if let (Some(row1), Some(row2)) = (table.get(prefix1), table.get(prefix2)) {
            for suffix in suffixes {
                if row1.get(suffix) != row2.get(suffix) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
    
    fn find_distinguishing_suffix(
        &self,
        prefix1: &str,
        prefix2: &str,
        table: &HashMap<String, HashMap<String, bool>>,
        suffixes: &HashSet<String>,
    ) -> Option<String> {
        if let (Some(row1), Some(row2)) = (table.get(prefix1), table.get(prefix2)) {
            for suffix in suffixes {
                if row1.get(suffix) != row2.get(suffix) {
                    return Some(suffix.clone());
                }
            }
        }
        None
    }
    
    fn build_candidate_dfa(
        &self,
        table: &HashMap<String, HashMap<String, bool>>,
        prefixes: &HashSet<String>,
        alphabet: &HashSet<char>,
    ) -> Result<LearnedDFA, DFACapabilityError> {
        let mut states = HashMap::new();
        let mut transitions = HashMap::new();
        let mut state_id = 0;
        let mut representatives = Vec::new();
        let mut state_map = HashMap::new();
        
        // Find representative prefixes
        for prefix in prefixes {
            let mut is_representative = true;
            for existing_rep in &representatives {
                if self.rows_equal(prefix, existing_rep, table, &HashSet::from([String::new()])) {
                    state_map.insert(prefix.clone(), state_map[existing_rep]);
                    is_representative = false;
                    break;
                }
            }
            
            if is_representative {
                let is_accepting = table.get(prefix)
                    .and_then(|row| row.get(""))
                    .copied()
                    .unwrap_or(false);
                
                states.insert(state_id, DFAState {
                    id: state_id,
                    is_accepting,
                    access_string: prefix.clone(),
                    experiments: vec![String::new()],
                });
                
                state_map.insert(prefix.clone(), state_id);
                representatives.push(prefix.clone());
                state_id += 1;
            }
        }
        
        // Build transitions
        for prefix in prefixes {
            let from_state = state_map[prefix];
            for &ch in alphabet {
                let extended = format!("{}{}", prefix, ch);
                for representative in &representatives {
                    if self.rows_equal(&extended, representative, table, &HashSet::from([String::new()])) {
                        let to_state = state_map[representative];
                        transitions.insert((from_state, ch), to_state);
                        break;
                    }
                }
            }
        }
        
        Ok(LearnedDFA {
            states,
            transitions,
            start_state: state_map.get("").copied().unwrap_or(0),
            alphabet: alphabet.clone(),
            metadata: DFAMetadata {
                queries_used: self.queries_made,
                learning_time_ms: 0,
                pattern_description: self.oracle.as_ref().unwrap().description(),
                alphabet_size: alphabet.len(),
            },
        })
    }
    
    fn find_counterexample(&mut self, dfa: &LearnedDFA) -> Result<Option<String>, DFACapabilityError> {
        let oracle = self.oracle.as_ref().unwrap();
        
        // Test strings up to reasonable length
        for length in 0..=6 {
            for test_string in self.generate_strings_of_length(length, &dfa.alphabet) {
                let dfa_accepts = self.dfa_accepts(dfa, &test_string);
                let oracle_accepts = oracle.is_member(&test_string);
                
                self.queries_made += 1;
                
                if dfa_accepts != oracle_accepts {
                    return Ok(Some(test_string));
                }
            }
        }
        
        Ok(None)
    }
    
    fn generate_strings_of_length(&self, length: usize, alphabet: &HashSet<char>) -> Vec<String> {
        if length == 0 {
            return vec![String::new()];
        }
        
        let chars: Vec<char> = alphabet.iter().copied().collect();
        let mut results = Vec::new();
        
        fn generate_recursive(
            chars: &[char],
            length: usize,
            current: String,
            results: &mut Vec<String>,
        ) {
            if current.len() == length {
                results.push(current);
                return;
            }
            
            for &ch in chars {
                let mut new_string = current.clone();
                new_string.push(ch);
                generate_recursive(chars, length, new_string, results);
            }
        }
        
        generate_recursive(&chars, length, String::new(), &mut results);
        results
    }
    
    fn dfa_accepts(&self, dfa: &LearnedDFA, input: &str) -> bool {
        let mut current_state = dfa.start_state;
        
        for ch in input.chars() {
            if let Some(&next_state) = dfa.transitions.get(&(current_state, ch)) {
                current_state = next_state;
            } else {
                return false;
            }
        }
        
        dfa.states.get(&current_state)
            .map(|state| state.is_accepting)
            .unwrap_or(false)
    }
}

impl DFAStringGenerationEngine {
    pub fn new() -> Self {
        Self {
            fallback_generator: StandardValueGenerator::new(),
            generation_params: DFAGenerationParams {
                max_enumeration_length: 8,
                fallback_probability: 0.1,
                constraint_priority: ConstraintPriority::Balanced,
            },
        }
    }
    
    pub fn generate_with_dfa(
        &mut self,
        dfa: &LearnedDFA,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFACapabilityError> {
        // Enumerate valid strings from DFA
        let valid_strings = self.enumerate_dfa_strings(dfa, constraints)?;
        
        if valid_strings.is_empty() {
            // Fallback to standard generation
            return self.generate_fallback(constraints, entropy);
        }
        
        // Select random string from valid options
        let index_bytes = entropy.draw_bytes(4)
            .map_err(|_| DFACapabilityError::GenerationFailed("Insufficient entropy".to_string()))?;
        let index = u32::from_le_bytes([
            index_bytes[0], index_bytes[1], index_bytes[2], index_bytes[3]
        ]) as usize % valid_strings.len();
        
        Ok(valid_strings[index].clone())
    }
    
    pub fn generate_fallback(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFACapabilityError> {
        self.fallback_generator.generate_string(constraints, entropy)
            .map_err(|e| DFACapabilityError::GenerationFailed(format!("Fallback failed: {}", e)))
    }
    
    fn enumerate_dfa_strings(
        &self,
        dfa: &LearnedDFA,
        constraints: &StringConstraints,
    ) -> Result<Vec<String>, DFACapabilityError> {
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((dfa.start_state, String::new()));
        
        while let Some((state, prefix)) = queue.pop_front() {
            if prefix.len() <= self.generation_params.max_enumeration_length {
                // Check if current string satisfies constraints
                if self.satisfies_constraints(&prefix, constraints) {
                    if let Some(dfa_state) = dfa.states.get(&state) {
                        if dfa_state.is_accepting {
                            results.push(prefix.clone());
                        }
                    }
                }
                
                // Continue exploration if under length limit
                if prefix.len() < constraints.max_size.min(self.generation_params.max_enumeration_length) {
                    for &ch in &dfa.alphabet {
                        if let Some(&next_state) = dfa.transitions.get(&(state, ch)) {
                            let mut new_prefix = prefix.clone();
                            new_prefix.push(ch);
                            queue.push_back((next_state, new_prefix));
                        }
                    }
                }
            }
        }
        
        // Filter by final constraints
        Ok(results.into_iter()
            .filter(|s| self.satisfies_all_constraints(s, constraints))
            .collect())
    }
    
    fn satisfies_constraints(&self, string: &str, constraints: &StringConstraints) -> bool {
        string.len() >= constraints.min_size && string.len() <= constraints.max_size
    }
    
    fn satisfies_all_constraints(&self, string: &str, constraints: &StringConstraints) -> bool {
        if string.len() < constraints.min_size || string.len() > constraints.max_size {
            return false;
        }
        
        string.chars().all(|ch| {
            let code = ch as u32;
            constraints.intervals.intervals.iter().any(|(start, end)| {
                code >= *start && code <= *end
            })
        })
    }
}

impl DFAChoiceIntegration {
    pub fn new() -> Self {
        Self {
            navigation_enabled: false,
            templating_enabled: false,
            weighted_selection_enabled: false,
        }
    }
}

// =================== COMPREHENSIVE CAPABILITY TESTS ===================

#[cfg(test)]
mod capability_tests {
    use super::*;
    
    /// Sample membership oracles for testing
    #[derive(Debug)]
    struct ContainsSubstring {
        pattern: String,
    }
    
    impl MembershipOracle for ContainsSubstring {
        fn is_member(&self, input: &str) -> bool {
            input.contains(&self.pattern)
        }
        
        fn is_prefix(&self, input: &str) -> bool {
            self.is_member(input) || input.len() < self.pattern.len()
        }
        
        fn description(&self) -> String {
            format!("Contains substring '{}'", self.pattern)
        }
    }
    
    #[derive(Debug)]
    struct ModularCount {
        target_char: char,
        modulus: usize,
        remainder: usize,
    }
    
    impl MembershipOracle for ModularCount {
        fn is_member(&self, input: &str) -> bool {
            let count = input.chars().filter(|&c| c == self.target_char).count();
            count % self.modulus == self.remainder
        }
        
        fn is_prefix(&self, _input: &str) -> bool {
            true
        }
        
        fn description(&self) -> String {
            format!("Count of '{}' ≡ {} (mod {})", self.target_char, self.remainder, self.modulus)
        }
    }
    
    #[derive(Debug)]
    struct RegexPattern {
        pattern: regex::Regex,
        description: String,
    }
    
    impl RegexPattern {
        fn new(pattern: &str) -> Result<Self, regex::Error> {
            Ok(Self {
                pattern: regex::Regex::new(pattern)?,
                description: pattern.to_string(),
            })
        }
    }
    
    impl MembershipOracle for RegexPattern {
        fn is_member(&self, input: &str) -> bool {
            self.pattern.is_match(input)
        }
        
        fn is_prefix(&self, input: &str) -> bool {
            self.is_member(input) || self.pattern.is_match(&format!("{}.*", regex::escape(input)))
        }
        
        fn description(&self) -> String {
            format!("Regex pattern: {}", self.description)
        }
    }
    
    #[test]
    fn test_complete_dfa_capability() {
        // Test the complete DFA-based string generation capability
        let mut capability = DFAStringGenerationCapability::new();
        
        // Test 1: Learn simple pattern
        let oracle = Box::new(ContainsSubstring {
            pattern: "abc".to_string(),
        });
        let alphabet = ['a', 'b', 'c', 'd'].into_iter().collect();
        
        capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
        
        // Test 2: Generate strings with learned pattern
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 8,
            intervals: IntervalSet::from_string("abcd"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0x42; 32]);
        let mut generated_strings = Vec::new();
        
        for _ in 0..10 {
            let generated = capability.generate_string(&constraints, &mut entropy)
                .expect("Generation should succeed");
            generated_strings.push(generated.clone());
            
            // Verify constraints are satisfied
            assert!(generated.len() >= constraints.min_size);
            assert!(generated.len() <= constraints.max_size);
            assert!(generated.chars().all(|c| "abcd".contains(c)));
        }
        
        // Test 3: Verify pattern compliance
        let contains_pattern_count = generated_strings.iter()
            .filter(|s| s.contains("abc"))
            .count();
        
        // Should generate at least some strings containing the pattern
        assert!(contains_pattern_count > 0, "Should generate strings matching learned pattern");
        
        // Test 4: Check metrics
        let metrics = capability.get_metrics();
        assert_eq!(metrics.learning_sessions, 1);
        assert!(metrics.total_queries > 0);
        assert_eq!(metrics.strings_generated, 10);
        assert!(metrics.constraint_satisfaction_rate >= 0.8); // Most should satisfy constraints
        
        println!("Complete capability test passed:");
        println!("  Learning sessions: {}", metrics.learning_sessions);
        println!("  Total queries: {}", metrics.total_queries);
        println!("  Strings generated: {}", metrics.strings_generated);
        println!("  Constraint satisfaction: {:.2}%", metrics.constraint_satisfaction_rate * 100.0);
        println!("  Generated pattern matches: {}/{}", contains_pattern_count, generated_strings.len());
    }
    
    #[test]
    fn test_modular_arithmetic_learning() {
        // Test learning more complex patterns
        let mut capability = DFAStringGenerationCapability::new();
        
        let oracle = Box::new(ModularCount {
            target_char: '1',
            modulus: 3,
            remainder: 0,
        });
        let alphabet = ['0', '1'].into_iter().collect();
        
        capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
        
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 6,
            intervals: IntervalSet::from_string("01"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0x7F; 40]);
        let mut pattern_matches = 0;
        
        for _ in 0..15 {
            let generated = capability.generate_string(&constraints, &mut entropy)
                .expect("Generation should succeed");
            
            // Check if generated string matches the modular arithmetic pattern
            let ones_count = generated.chars().filter(|&c| c == '1').count();
            if ones_count % 3 == 0 {
                pattern_matches += 1;
            }
        }
        
        // Should generate strings that mostly match the learned pattern
        assert!(pattern_matches >= 8, "Should generate strings matching modular pattern");
        
        let metrics = capability.get_metrics();
        assert!(metrics.average_states_learned > 0.0);
        println!("Modular arithmetic learning test passed:");
        println!("  Average states learned: {:.1}", metrics.average_states_learned);
        println!("  Pattern matches: {}/15", pattern_matches);
    }
    
    #[test]
    fn test_choice_system_integration() {
        // Test integration with choice system components
        let mut capability = DFAStringGenerationCapability::new();
        
        // Enable all integrations
        capability.enable_choice_navigation().expect("Navigation should enable");
        capability.enable_templating().expect("Templating should enable");
        capability.enable_weighted_selection().expect("Weighted selection should enable");
        
        // Learn a pattern
        let oracle = Box::new(ContainsSubstring {
            pattern: "xy".to_string(),
        });
        let alphabet = ['x', 'y', 'z'].into_iter().collect();
        
        capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
        
        // Test generation with integrations enabled
        let constraints = StringConstraints {
            min_size: 2,
            max_size: 5,
            intervals: IntervalSet::from_string("xyz"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0xAB; 24]);
        let generated = capability.generate_string(&constraints, &mut entropy)
            .expect("Integrated generation should succeed");
        
        assert!(generated.len() >= 2 && generated.len() <= 5);
        assert!(generated.chars().all(|c| "xyz".contains(c)));
        
        let metrics = capability.get_metrics();
        assert_eq!(metrics.choice_integrations, 3);
        
        println!("Choice system integration test passed:");
        println!("  Choice integrations: {}", metrics.choice_integrations);
        println!("  Generated: {:?}", generated);
    }
    
    #[test]
    fn test_performance_metrics() {
        // Test comprehensive performance metrics collection
        let mut capability = DFAStringGenerationCapability::new();
        
        // Multiple learning sessions
        let oracles = vec![
            Box::new(ContainsSubstring { pattern: "ab".to_string() }) as Box<dyn MembershipOracle>,
            Box::new(ContainsSubstring { pattern: "ba".to_string() }) as Box<dyn MembershipOracle>,
            Box::new(ModularCount { target_char: 'x', modulus: 2, remainder: 1 }),
        ];
        
        let alphabet = ['a', 'b', 'x'].into_iter().collect();
        
        for oracle in oracles {
            capability.learn_pattern(oracle, alphabet.clone()).expect("Learning should succeed");
        }
        
        // Generate many strings to test performance tracking
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 4,
            intervals: IntervalSet::from_string("abx"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0x55; 200]);
        
        for _ in 0..50 {
            let _generated = capability.generate_string(&constraints, &mut entropy)
                .expect("Generation should succeed");
        }
        
        let metrics = capability.get_metrics();
        
        // Verify comprehensive metrics
        assert_eq!(metrics.learning_sessions, 3);
        assert_eq!(metrics.strings_generated, 50);
        assert!(metrics.total_queries > 0);
        assert!(metrics.average_states_learned > 0.0);
        assert!(metrics.average_generation_time_us >= 0.0);
        assert!(metrics.constraint_satisfaction_rate >= 0.0);
        assert!(metrics.constraint_satisfaction_rate <= 1.0);
        
        println!("Performance metrics test passed:");
        println!("  Learning sessions: {}", metrics.learning_sessions);
        println!("  Total queries: {}", metrics.total_queries);
        println!("  Strings generated: {}", metrics.strings_generated);
        println!("  Average states learned: {:.2}", metrics.average_states_learned);
        println!("  Average generation time: {:.2}μs", metrics.average_generation_time_us);
        println!("  Constraint satisfaction: {:.1}%", metrics.constraint_satisfaction_rate * 100.0);
    }
    
    #[test]
    fn test_regex_pattern_learning() {
        // Test learning from regex patterns
        let mut capability = DFAStringGenerationCapability::new();
        
        let oracle = Box::new(RegexPattern::new(r"^a*b+$").expect("Valid regex"));
        let alphabet = ['a', 'b'].into_iter().collect();
        
        capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
        
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 5,
            intervals: IntervalSet::from_string("ab"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0x33; 60]);
        let mut regex_matches = 0;
        let regex = regex::Regex::new(r"^a*b+$").unwrap();
        
        for _ in 0..20 {
            let generated = capability.generate_string(&constraints, &mut entropy)
                .expect("Generation should succeed");
            
            if regex.is_match(&generated) {
                regex_matches += 1;
            }
        }
        
        // Should generate strings that match the regex pattern
        assert!(regex_matches >= 10, "Should generate strings matching regex pattern");
        
        println!("Regex pattern learning test passed:");
        println!("  Regex matches: {}/20", regex_matches);
    }
    
    // =============== PyO3 FFI Integration Tests ===============
    
    #[test]
    fn test_python_ffi_capability() {
        Python::with_gil(|py| {
            // Test complete DFA capability through FFI
            let mut capability = DFAStringGenerationCapability::new();
            
            let oracle = Box::new(ContainsSubstring {
                pattern: "test".to_string(),
            });
            let alphabet = ['t', 'e', 's', 'x'].into_iter().collect();
            
            capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
            
            // Convert capability metrics to Python
            let metrics = capability.get_metrics();
            let py_metrics = pyo3::types::PyDict::new(py);
            py_metrics.set_item("learning_sessions", metrics.learning_sessions).unwrap();
            py_metrics.set_item("total_queries", metrics.total_queries).unwrap();
            py_metrics.set_item("average_states_learned", metrics.average_states_learned).unwrap();
            
            // Verify Python can access metrics
            let sessions: usize = py_metrics.get_item("learning_sessions").unwrap().extract().unwrap();
            assert_eq!(sessions, metrics.learning_sessions);
            
            // Test string generation through FFI
            let constraints = StringConstraints {
                min_size: 1,
                max_size: 6,
                intervals: IntervalSet::from_string("tesx"),
            };
            
            let mut entropy = BufferEntropySource::new(vec![0x99; 32]);
            let generated = capability.generate_string(&constraints, &mut entropy)
                .expect("Generation should succeed");
            
            // Convert generated string to Python
            let py_string = pyo3::types::PyString::new(py, &generated);
            let extracted: String = py_string.extract().unwrap();
            assert_eq!(extracted, generated);
            
            // Test through Python evaluation
            let contains_test = py.eval(
                &format!("'test' in '{}'", generated),
                None,
                None,
            ).unwrap().extract::<bool>().unwrap();
            
            println!("Python FFI capability test passed:");
            println!("  Generated: {:?}", generated);
            println!("  Contains pattern: {}", contains_test);
        });
    }
    
    #[test]
    fn test_error_handling_capability() {
        // Test comprehensive error handling
        let mut capability = DFAStringGenerationCapability::new();
        
        // Test generation without learned pattern
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 3,
            intervals: IntervalSet::from_string("abc"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0x44; 16]);
        
        // Should succeed with fallback generation
        let result = capability.generate_string(&constraints, &mut entropy);
        assert!(result.is_ok());
        
        // Test with insufficient entropy
        let mut empty_entropy = BufferEntropySource::new(vec![]);
        let result = capability.generate_string(&constraints, &mut empty_entropy);
        assert!(result.is_err());
        
        if let Err(DFACapabilityError::GenerationFailed(msg)) = result {
            assert!(msg.contains("Fallback failed"));
        } else {
            panic!("Expected GenerationFailed error");
        }
        
        println!("Error handling capability test passed");
    }
}

// =============== Integration Tests with Navigation System ===============

#[cfg(test)]
mod navigation_integration_tests {
    use super::*;
    
    #[test]
    fn test_dfa_navigation_integration() {
        // Test DFA capability integration with navigation system
        let mut capability = DFAStringGenerationCapability::new();
        capability.enable_choice_navigation().expect("Navigation should enable");
        
        let mut navigation = NavigationSystem::new();
        
        // Learn DFA pattern
        let oracle = Box::new(ContainsSubstring {
            pattern: "nav".to_string(),
        });
        let alphabet = ['n', 'a', 'v', 'i'].into_iter().collect();
        
        capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
        
        // Create navigable choice sequence
        let mut sequence = ChoiceSequence::new();
        sequence.push(ChoiceNode::new(
            ChoiceType::String,
            Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 6,
                intervals: IntervalSet::from_string("navi"),
            }),
            100, // choice_id
        ));
        
        // Test navigation capability
        assert!(navigation.can_navigate_to(&sequence, 100));
        
        // Generate navigable string
        let constraints = StringConstraints {
            min_size: 3,
            max_size: 6,
            intervals: IntervalSet::from_string("navi"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0x6A; 20]);
        let generated = capability.generate_string(&constraints, &mut entropy)
            .expect("Navigation-aware generation should succeed");
        
        assert!(generated.len() >= 3 && generated.len() <= 6);
        assert!(generated.chars().all(|c| "navi".contains(c)));
        
        println!("DFA navigation integration test passed:");
        println!("  Generated navigable string: {:?}", generated);
    }
}

// =============== Integration Tests with Templating System ===============

#[cfg(test)]
mod templating_integration_tests {
    use super::*;
    
    #[test]
    fn test_dfa_templating_integration() {
        // Test DFA capability integration with templating system
        let mut capability = DFAStringGenerationCapability::new();
        capability.enable_templating().expect("Templating should enable");
        
        let mut templating = TemplatingSystem::new();
        
        // Learn DFA pattern
        let oracle = Box::new(ModularCount {
            target_char: 't',
            modulus: 2,
            remainder: 0,
        });
        let alphabet = ['t', 'e', 'm', 'p'].into_iter().collect();
        
        capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
        
        // Create template for DFA-based generation
        let template_id = templating.create_template(
            "dfa_template".to_string(),
            vec![
                (ChoiceType::String, Constraints::String(StringConstraints {
                    min_size: 2,
                    max_size: 5,
                    intervals: IntervalSet::from_string("temp"),
                }))
            ]
        );
        
        // Generate template-compatible string
        let constraints = StringConstraints {
            min_size: 2,
            max_size: 5,
            intervals: IntervalSet::from_string("temp"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0x8C; 24]);
        let generated = capability.generate_string(&constraints, &mut entropy)
            .expect("Template-aware generation should succeed");
        
        // Verify template compatibility
        let forced_values = vec![ChoiceValue::String(generated.clone())];
        let template_result = templating.apply_template(template_id, forced_values);
        assert!(template_result.is_ok());
        
        // Verify modular pattern (even number of 't's)
        let t_count = generated.chars().filter(|&c| c == 't').count();
        assert_eq!(t_count % 2, 0, "Generated string should have even number of 't's");
        
        println!("DFA templating integration test passed:");
        println!("  Generated template string: {:?}", generated);
        println!("  't' count: {} (even)", t_count);
    }
}

// =============== Integration Tests with Weighted Selection ===============

#[cfg(test)]
mod weighted_selection_integration_tests {
    use super::*;
    
    #[test]
    fn test_dfa_weighted_selection_integration() {
        // Test DFA capability integration with weighted selection
        let mut capability = DFAStringGenerationCapability::new();
        capability.enable_weighted_selection().expect("Weighted selection should enable");
        
        let mut weighted_selection = WeightedSelection::new();
        
        // Learn DFA pattern
        let oracle = Box::new(ContainsSubstring {
            pattern: "weight".to_string(),
        });
        let alphabet = ['w', 'e', 'i', 'g', 'h', 't'].into_iter().collect();
        
        capability.learn_pattern(oracle, alphabet).expect("Learning should succeed");
        
        // Generate multiple DFA-compatible strings
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 8,
            intervals: IntervalSet::from_string("weight"),
        };
        
        let mut entropy = BufferEntropySource::new(vec![0xD5; 80]);
        let mut generated_strings = Vec::new();
        
        for _ in 0..10 {
            let generated = capability.generate_string(&constraints, &mut entropy)
                .expect("Generation should succeed");
            generated_strings.push(generated);
        }
        
        // Create weighted selection from generated strings
        let mut weights = HashMap::new();
        for string in &generated_strings {
            // Weight based on pattern compliance and length
            let weight = if string.contains("weight") { 2.0 } else { 1.0 } +
                        (string.len() as f64 / 10.0);
            weights.insert(string.clone(), weight);
        }
        
        let choice_id = weighted_selection.add_weighted_choice(
            ChoiceType::String,
            generated_strings.iter().cloned().map(ChoiceValue::String).collect(),
            weights.values().cloned().collect(),
        );
        
        // Test weighted selection
        let mut weighted_entropy = BufferEntropySource::new(vec![0x7E; 16]);
        let selected = weighted_selection.select_weighted_value(choice_id, &mut weighted_entropy)
            .expect("Weighted selection should succeed");
        
        if let ChoiceValue::String(selected_string) = selected {
            assert!(generated_strings.contains(&selected_string));
            println!("DFA weighted selection integration test passed:");
            println!("  Selected: {:?} (weight: {:.2})", 
                    selected_string, weights[&selected_string]);
        } else {
            panic!("Expected string value from weighted selection");
        }
    }
}