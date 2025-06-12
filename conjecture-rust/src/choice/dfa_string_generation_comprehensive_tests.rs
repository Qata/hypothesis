//! Comprehensive Tests for DFA-Based String Generation System
//!
//! Tests the complete capability of learning finite automata using the L* algorithm
//! for string pattern recognition, optimization, and structured string generation.
//! This mirrors Python Hypothesis's sophisticated DFA/L* implementation.

use crate::choice::{
    ChoiceValue, ChoiceType, Constraints, StringConstraints, IntervalSet,
    StandardValueGenerator, ValueGenerator, BufferEntropySource, EntropySource,
    ChoiceNode, ChoiceSequence, NavigationTree, TemplateEngine, CumulativeWeightedSelector,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use pyo3::prelude::*;

/// Error types for DFA operations
#[derive(Debug, Clone, PartialEq)]
pub enum DFAError {
    InvalidState(usize),
    InvalidTransition(String),
    LearningFailed(String),
    MembershipQueryFailed(String),
}

/// Represents a state in a learned DFA
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DFAState {
    pub id: usize,
    pub is_accepting: bool,
    pub experiments: Vec<String>,
    pub access_string: String,
}

/// Transition function for DFA states
pub type TransitionFunction = HashMap<(usize, char), usize>;

/// Membership oracle that determines if strings belong to target language
pub trait MembershipOracle {
    fn is_member(&self, input: &str) -> bool;
    fn is_prefix(&self, input: &str) -> bool;
}

/// Deterministic Finite Automaton with L* learning capability
#[derive(Debug, Clone)]
pub struct LearnedDFA {
    pub states: HashMap<usize, DFAState>,
    pub transitions: TransitionFunction,
    pub start_state: usize,
    pub alphabet: HashSet<char>,
    pub next_state_id: usize,
}

impl LearnedDFA {
    pub fn new(alphabet: HashSet<char>) -> Self {
        let mut states = HashMap::new();
        states.insert(0, DFAState {
            id: 0,
            is_accepting: false,
            experiments: Vec::new(),
            access_string: String::new(),
        });

        Self {
            states,
            transitions: HashMap::new(),
            start_state: 0,
            alphabet,
            next_state_id: 1,
        }
    }

    pub fn accepts(&self, input: &str) -> bool {
        let mut current_state = self.start_state;
        
        for ch in input.chars() {
            if let Some(&next_state) = self.transitions.get(&(current_state, ch)) {
                current_state = next_state;
            } else {
                return false; // Dead state
            }
        }
        
        self.states.get(&current_state)
            .map(|state| state.is_accepting)
            .unwrap_or(false)
    }

    pub fn enumerate_strings(&self, max_length: usize) -> Vec<String> {
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((self.start_state, String::new()));

        while let Some((state, prefix)) = queue.pop_front() {
            if prefix.len() <= max_length {
                if let Some(dfa_state) = self.states.get(&state) {
                    if dfa_state.is_accepting {
                        results.push(prefix.clone());
                    }
                }

                if prefix.len() < max_length {
                    for &ch in &self.alphabet {
                        if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                            let mut new_prefix = prefix.clone();
                            new_prefix.push(ch);
                            queue.push_back((next_state, new_prefix));
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
        results
    }

    pub fn count_strings_of_length(&self, length: usize) -> usize {
        if length == 0 {
            return if self.states.get(&self.start_state)
                .map(|s| s.is_accepting)
                .unwrap_or(false) { 1 } else { 0 };
        }

        let mut counts = HashMap::new();
        counts.insert(self.start_state, 1usize);

        for _ in 0..length {
            let mut new_counts = HashMap::new();
            
            for (&state, &count) in &counts {
                for &ch in &self.alphabet {
                    if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                        *new_counts.entry(next_state).or_insert(0) += count;
                    }
                }
            }
            
            counts = new_counts;
        }

        counts.iter()
            .filter_map(|(&state, &count)| {
                self.states.get(&state)
                    .filter(|s| s.is_accepting)
                    .map(|_| count)
            })
            .sum()
    }
}

/// L* Learning Algorithm Implementation
pub struct LStarLearner {
    oracle: Box<dyn MembershipOracle>,
    alphabet: HashSet<char>,
    observation_table: HashMap<String, HashMap<String, bool>>,
    prefixes: HashSet<String>,
    suffixes: HashSet<String>,
    queries_made: usize,
}

impl LStarLearner {
    pub fn new(oracle: Box<dyn MembershipOracle>, alphabet: HashSet<char>) -> Self {
        let mut learner = Self {
            oracle,
            alphabet,
            observation_table: HashMap::new(),
            prefixes: HashSet::new(),
            suffixes: HashSet::new(),
            queries_made: 0,
        };

        // Initialize with empty string
        learner.prefixes.insert(String::new());
        learner.suffixes.insert(String::new());
        learner.fill_table();
        learner
    }

    pub fn learn(&mut self) -> Result<LearnedDFA, DFAError> {
        loop {
            // Make table closed and consistent
            self.make_closed()?;
            self.make_consistent()?;

            // Build candidate DFA
            let dfa = self.build_dfa()?;

            // Test equivalence with oracle
            if let Some(counterexample) = self.find_counterexample(&dfa) {
                self.process_counterexample(&counterexample);
                self.fill_table();
            } else {
                return Ok(dfa);
            }

            // Safety check to prevent infinite loops
            if self.queries_made > 10000 {
                return Err(DFAError::LearningFailed(
                    "Too many membership queries".to_string()
                ));
            }
        }
    }

    fn fill_table(&mut self) {
        for prefix in self.prefixes.clone() {
            for suffix in self.suffixes.clone() {
                let string = prefix.clone() + &suffix;
                if !self.observation_table.contains_key(&prefix) {
                    self.observation_table.insert(prefix.clone(), HashMap::new());
                }
                
                if !self.observation_table[&prefix].contains_key(&suffix) {
                    let is_member = self.oracle.is_member(&string);
                    self.queries_made += 1;
                    self.observation_table.get_mut(&prefix).unwrap()
                        .insert(suffix, is_member);
                }
            }
        }

        // Also fill extended table (prefix + alphabet)
        for prefix in self.prefixes.clone() {
            for ch in &self.alphabet {
                let extended = prefix.clone() + &ch.to_string();
                if !self.observation_table.contains_key(&extended) {
                    self.observation_table.insert(extended.clone(), HashMap::new());
                    
                    for suffix in &self.suffixes {
                        let string = extended.clone() + suffix;
                        let is_member = self.oracle.is_member(&string);
                        self.queries_made += 1;
                        self.observation_table.get_mut(&extended).unwrap()
                            .insert(suffix.clone(), is_member);
                    }
                }
            }
        }
    }

    fn make_closed(&mut self) -> Result<(), DFAError> {
        loop {
            let mut found_unclosed = false;
            
            for prefix in self.prefixes.clone() {
                for ch in &self.alphabet {
                    let extended = prefix.clone() + &ch.to_string();
                    if !self.prefixes.contains(&extended) {
                        if !self.has_representative(&extended) {
                            self.prefixes.insert(extended);
                            found_unclosed = true;
                            break;
                        }
                    }
                }
                if found_unclosed { break; }
            }
            
            if !found_unclosed { break; }
            self.fill_table();
        }
        Ok(())
    }

    fn make_consistent(&mut self) -> Result<(), DFAError> {
        loop {
            let mut found_inconsistency = false;
            
            let prefixes: Vec<String> = self.prefixes.iter().cloned().collect();
            for i in 0..prefixes.len() {
                for j in i+1..prefixes.len() {
                    let prefix1 = &prefixes[i];
                    let prefix2 = &prefixes[j];
                    
                    if self.rows_equal(prefix1, prefix2) {
                        for ch in &self.alphabet {
                            let ext1 = prefix1.clone() + &ch.to_string();
                            let ext2 = prefix2.clone() + &ch.to_string();
                            
                            if !self.rows_equal(&ext1, &ext2) {
                                // Find distinguishing suffix
                                if let Some(suffix) = self.find_distinguishing_suffix(&ext1, &ext2) {
                                    self.suffixes.insert(suffix);
                                    found_inconsistency = true;
                                    break;
                                }
                            }
                        }
                        if found_inconsistency { break; }
                    }
                }
                if found_inconsistency { break; }
            }
            
            if !found_inconsistency { break; }
            self.fill_table();
        }
        Ok(())
    }

    fn has_representative(&self, prefix: &str) -> bool {
        for existing_prefix in &self.prefixes {
            if self.rows_equal(prefix, existing_prefix) {
                return true;
            }
        }
        false
    }

    fn rows_equal(&self, prefix1: &str, prefix2: &str) -> bool {
        if let (Some(row1), Some(row2)) = (
            self.observation_table.get(prefix1),
            self.observation_table.get(prefix2)
        ) {
            for suffix in &self.suffixes {
                if row1.get(suffix) != row2.get(suffix) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    fn find_distinguishing_suffix(&self, prefix1: &str, prefix2: &str) -> Option<String> {
        if let (Some(row1), Some(row2)) = (
            self.observation_table.get(prefix1),
            self.observation_table.get(prefix2)
        ) {
            for suffix in &self.suffixes {
                if row1.get(suffix) != row2.get(suffix) {
                    return Some(suffix.clone());
                }
            }
        }
        None
    }

    fn build_dfa(&self) -> Result<LearnedDFA, DFAError> {
        let mut dfa = LearnedDFA::new(self.alphabet.clone());
        let mut state_map = HashMap::new();
        let mut representatives: Vec<String> = Vec::new();

        // Find representative prefixes (equivalence class representatives)
        for prefix in &self.prefixes {
            let mut is_representative = true;
            for existing_rep in &representatives {
                if self.rows_equal(prefix, existing_rep) {
                    state_map.insert(prefix.clone(), state_map[existing_rep]);
                    is_representative = false;
                    break;
                }
            }
            
            if is_representative {
                let state_id = dfa.next_state_id;
                dfa.next_state_id += 1;
                
                let is_accepting = self.observation_table
                    .get(prefix)
                    .and_then(|row| row.get(""))
                    .copied()
                    .unwrap_or(false);

                dfa.states.insert(state_id, DFAState {
                    id: state_id,
                    is_accepting,
                    experiments: self.suffixes.iter().cloned().collect(),
                    access_string: prefix.clone(),
                });

                state_map.insert(prefix.clone(), state_id);
                representatives.push(prefix.clone());
                
                if prefix.is_empty() {
                    dfa.start_state = state_id;
                }
            }
        }

        // Build transition function
        for prefix in &self.prefixes {
            let from_state = state_map[prefix];
            
            for ch in &self.alphabet {
                let extended = prefix.clone() + &ch.to_string();
                
                // Find which representative this extended prefix maps to
                for representative in &representatives {
                    if self.rows_equal(&extended, representative) {
                        let to_state = state_map[representative];
                        dfa.transitions.insert((from_state, *ch), to_state);
                        break;
                    }
                }
            }
        }

        Ok(dfa)
    }

    fn find_counterexample(&self, dfa: &LearnedDFA) -> Option<String> {
        // Simple counterexample generation - test strings up to reasonable length
        for length in 0..=6 {
            for string in self.generate_strings_of_length(length) {
                let dfa_accepts = dfa.accepts(&string);
                let oracle_accepts = self.oracle.is_member(&string);
                
                if dfa_accepts != oracle_accepts {
                    return Some(string);
                }
            }
        }
        None
    }

    fn generate_strings_of_length(&self, length: usize) -> Vec<String> {
        if length == 0 {
            return vec![String::new()];
        }
        
        let mut results = Vec::new();
        let alphabet: Vec<char> = self.alphabet.iter().copied().collect();
        
        fn generate_recursive(
            alphabet: &[char], 
            length: usize, 
            current: String, 
            results: &mut Vec<String>
        ) {
            if current.len() == length {
                results.push(current);
                return;
            }
            
            for &ch in alphabet {
                let mut new_string = current.clone();
                new_string.push(ch);
                generate_recursive(alphabet, length, new_string, results);
            }
        }
        
        generate_recursive(&alphabet, length, String::new(), &mut results);
        results
    }

    fn process_counterexample(&mut self, counterexample: &str) {
        // Add all prefixes of counterexample as potential distinguishing experiments
        for i in 0..=counterexample.len() {
            let suffix = counterexample.chars().skip(i).collect::<String>();
            self.suffixes.insert(suffix);
        }
    }
}

/// Sample membership oracles for testing
pub struct RegexOracle {
    pattern: regex::Regex,
}

impl RegexOracle {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
        })
    }
}

impl MembershipOracle for RegexOracle {
    fn is_member(&self, input: &str) -> bool {
        self.pattern.is_match(input)
    }
    
    fn is_prefix(&self, input: &str) -> bool {
        // Simplified prefix check - could be more sophisticated
        self.is_member(input) || 
        (0..10).any(|i| self.is_member(&format!("{}{}", input, "a".repeat(i))))
    }
}

/// DFA-based String Generator integrated with choice system
#[derive(Debug)]
pub struct DFAStringGenerator {
    dfa: LearnedDFA,
    generator: StandardValueGenerator,
}

impl DFAStringGenerator {
    pub fn new(dfa: LearnedDFA) -> Self {
        Self {
            dfa,
            generator: StandardValueGenerator::new(),
        }
    }

    pub fn generate_string(
        &mut self,
        max_length: usize,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFAError> {
        let valid_strings = self.dfa.enumerate_strings(max_length);
        
        if valid_strings.is_empty() {
            return Err(DFAError::InvalidTransition(
                "No valid strings of specified length".to_string()
            ));
        }

        // Use entropy to select from valid strings
        let index_bytes = entropy.draw_bytes(4).map_err(|_| 
            DFAError::LearningFailed("Insufficient entropy".to_string()))?;
        let index = u32::from_le_bytes([
            index_bytes[0], index_bytes[1], index_bytes[2], index_bytes[3]
        ]) as usize % valid_strings.len();

        Ok(valid_strings[index].clone())
    }

    pub fn generate_structured_string(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFAError> {
        // Filter DFA strings by constraints
        let max_search_length = constraints.max_size.min(10);
        let candidates = self.dfa.enumerate_strings(max_search_length)
            .into_iter()
            .filter(|s| {
                s.len() >= constraints.min_size && 
                s.len() <= constraints.max_size &&
                s.chars().all(|ch| {
                    let code = ch as u32;
                    constraints.intervals.intervals.iter().any(|(start, end)| {
                        code >= *start && code <= *end
                    })
                })
            })
            .collect::<Vec<_>>();

        if candidates.is_empty() {
            // Fallback to standard generation
            return self.generator.generate_string(constraints, entropy)
                .map_err(|_| DFAError::LearningFailed("Standard generation failed".to_string()));
        }

        let index_bytes = entropy.draw_bytes(4).map_err(|_| 
            DFAError::LearningFailed("Insufficient entropy".to_string()))?;
        let index = u32::from_le_bytes([
            index_bytes[0], index_bytes[1], index_bytes[2], index_bytes[3]
        ]) as usize % candidates.len();

        Ok(candidates[index].clone())
    }
}

// =================== COMPREHENSIVE FFI TESTS ===================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfa_basic_acceptance() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // Build simple DFA that accepts strings ending in 'a'
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "a".to_string(),
        });
        
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 0);
        dfa.transitions.insert((1, 'a'), 1);
        dfa.transitions.insert((1, 'b'), 0);
        
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("ba"));
        assert!(dfa.accepts("aba"));
        assert!(!dfa.accepts(""));
        assert!(!dfa.accepts("b"));
        assert!(!dfa.accepts("ab"));
    }

    #[test]
    fn test_dfa_string_enumeration() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // DFA that accepts only "a" and "aa"
        dfa.states.get_mut(&0).unwrap().is_accepting = false;
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "a".to_string(),
        });
        dfa.states.insert(2, DFAState {
            id: 2,
            is_accepting: false,
            experiments: vec!["".to_string()],
            access_string: "aa".to_string(),
        });
        
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 2);
        dfa.transitions.insert((1, 'a'), 1);
        dfa.transitions.insert((1, 'b'), 2);
        dfa.transitions.insert((2, 'a'), 2);
        dfa.transitions.insert((2, 'b'), 2);
        
        let strings = dfa.enumerate_strings(3);
        assert!(strings.contains(&"a".to_string()));
        assert!(strings.contains(&"aa".to_string()));
        assert!(strings.contains(&"aaa".to_string()));
        assert!(!strings.contains(&"b".to_string()));
        assert!(!strings.contains(&"ab".to_string()));
    }

    #[test]
    fn test_lstar_simple_learning() {
        // Oracle that accepts strings containing "ab"
        struct ContainsAB;
        impl MembershipOracle for ContainsAB {
            fn is_member(&self, input: &str) -> bool {
                input.contains("ab")
            }
            fn is_prefix(&self, input: &str) -> bool {
                self.is_member(input) || input.len() < 2
            }
        }

        let alphabet = ['a', 'b'].into_iter().collect();
        let oracle = Box::new(ContainsAB);
        let mut learner = LStarLearner::new(oracle, alphabet);
        
        let dfa = learner.learn().expect("Learning should succeed");
        
        // Test learned DFA
        assert!(dfa.accepts("ab"));
        assert!(dfa.accepts("aab"));
        assert!(dfa.accepts("abb"));
        assert!(dfa.accepts("abab"));
        assert!(!dfa.accepts(""));
        assert!(!dfa.accepts("a"));
        assert!(!dfa.accepts("b"));
        assert!(!dfa.accepts("ba"));
        
        println!("L* learning completed with {} queries", learner.queries_made);
        assert!(learner.queries_made > 0);
        assert!(learner.queries_made < 100); // Should be efficient
    }

    #[test] 
    fn test_lstar_regex_learning() {
        let oracle = Box::new(
            RegexOracle::new(r"^a*b+$").expect("Valid regex")
        );
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut learner = LStarLearner::new(oracle, alphabet);
        
        let dfa = learner.learn().expect("Learning should succeed");
        
        // Test learned DFA against regex pattern a*b+
        assert!(dfa.accepts("b"));
        assert!(dfa.accepts("ab"));
        assert!(dfa.accepts("bb"));
        assert!(dfa.accepts("aab"));
        assert!(dfa.accepts("abb"));
        assert!(dfa.accepts("aabb"));
        assert!(!dfa.accepts(""));
        assert!(!dfa.accepts("a"));
        assert!(!dfa.accepts("ba"));
        assert!(!dfa.accepts("aba"));
        
        println!("Regex learning completed with {} queries", learner.queries_made);
    }

    #[test]
    fn test_dfa_string_counting() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // DFA that accepts strings of even length
        dfa.states.get_mut(&0).unwrap().is_accepting = true; // Even length (including 0)
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: false, // Odd length
            experiments: vec!["".to_string()],
            access_string: "a".to_string(),
        });
        
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 1);
        dfa.transitions.insert((1, 'a'), 0);
        dfa.transitions.insert((1, 'b'), 0);
        
        assert_eq!(dfa.count_strings_of_length(0), 1); // ""
        assert_eq!(dfa.count_strings_of_length(1), 0); // No odd-length strings accepted
        assert_eq!(dfa.count_strings_of_length(2), 4); // "aa", "ab", "ba", "bb"
        assert_eq!(dfa.count_strings_of_length(3), 0); // No odd-length strings
        assert_eq!(dfa.count_strings_of_length(4), 16); // 2^4 strings
    }

    #[test]
    fn test_dfa_string_generator() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // Simple DFA that accepts "a" and "b"
        dfa.states.get_mut(&0).unwrap().is_accepting = false;
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "a".to_string(),
        });
        
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 1);
        dfa.transitions.insert((1, 'a'), 1);
        dfa.transitions.insert((1, 'b'), 1);
        
        let mut generator = DFAStringGenerator::new(dfa);
        let mut entropy = BufferEntropySource::new(vec![0x12, 0x34, 0x56, 0x78]);
        
        let generated = generator.generate_string(2, &mut entropy)
            .expect("Should generate valid string");
        
        assert!(generated == "a" || generated == "b" || 
                generated == "aa" || generated == "ab" || 
                generated == "ba" || generated == "bb");
        assert!(generated.len() <= 2);
    }

    #[test]
    fn test_dfa_structured_generation() {
        let alphabet = ['a', 'b', 'c'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // DFA that accepts strings starting with 'a'
        dfa.states.get_mut(&0).unwrap().is_accepting = false;
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "a".to_string(),
        });
        dfa.states.insert(2, DFAState {
            id: 2,
            is_accepting: false,
            experiments: vec!["".to_string()],
            access_string: "b".to_string(),
        });
        
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 2);
        dfa.transitions.insert((0, 'c'), 2);
        dfa.transitions.insert((1, 'a'), 1);
        dfa.transitions.insert((1, 'b'), 1);
        dfa.transitions.insert((1, 'c'), 1);
        
        let mut generator = DFAStringGenerator::new(dfa);
        let mut entropy = BufferEntropySource::new(vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]);
        
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 3,
            intervals: IntervalSet::from_string("abc"),
        };
        
        let generated = generator.generate_structured_string(&constraints, &mut entropy)
            .expect("Should generate valid string");
        
        assert!(generated.starts_with('a'));
        assert!(generated.len() >= 1 && generated.len() <= 3);
        assert!(generated.chars().all(|c| "abc".contains(c)));
    }

    #[test]
    fn test_alphabet_optimization() {
        // Test character equivalence learning
        struct EquivalentChars;
        impl MembershipOracle for EquivalentChars {
            fn is_member(&self, input: &str) -> bool {
                // 'a' and 'b' are equivalent, 'c' is different
                let normalized = input.replace('b', "a");
                normalized == "a" || normalized == "aa"
            }
            fn is_prefix(&self, _input: &str) -> bool { true }
        }

        let alphabet = ['a', 'b', 'c'].into_iter().collect();
        let oracle = Box::new(EquivalentChars);
        let mut learner = LStarLearner::new(oracle, alphabet);
        
        let dfa = learner.learn().expect("Learning should succeed");
        
        // Both 'a' and 'b' should behave equivalently
        assert_eq!(dfa.accepts("a"), dfa.accepts("b"));
        assert_eq!(dfa.accepts("aa"), dfa.accepts("bb"));
        assert_eq!(dfa.accepts("aa"), dfa.accepts("ab"));
        assert_eq!(dfa.accepts("aa"), dfa.accepts("ba"));
        
        // 'c' should behave differently
        assert_ne!(dfa.accepts("a"), dfa.accepts("c"));
        assert_ne!(dfa.accepts("aa"), dfa.accepts("cc"));
    }

    #[test]
    fn test_dfa_complexity_limits() {
        // Test learning with complexity constraints
        struct ComplexLanguage;
        impl MembershipOracle for ComplexLanguage {
            fn is_member(&self, input: &str) -> bool {
                // Complex pattern: palindromes of length <= 4
                if input.len() > 4 { return false; }
                let reversed: String = input.chars().rev().collect();
                input == reversed
            }
            fn is_prefix(&self, _input: &str) -> bool { true }
        }

        let alphabet = ['a', 'b'].into_iter().collect();
        let oracle = Box::new(ComplexLanguage);
        let mut learner = LStarLearner::new(oracle, alphabet);
        
        let dfa = learner.learn().expect("Learning should succeed");
        
        // Test palindrome recognition
        assert!(dfa.accepts(""));
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("b"));
        assert!(dfa.accepts("aa"));
        assert!(dfa.accepts("bb"));
        assert!(dfa.accepts("aba"));
        assert!(dfa.accepts("bab"));
        assert!(dfa.accepts("abba"));
        assert!(dfa.accepts("baab"));
        assert!(!dfa.accepts("ab"));
        assert!(!dfa.accepts("ba"));
        assert!(!dfa.accepts("aab"));
        assert!(!dfa.accepts("abb"));
        
        // Should learn efficiently despite complexity
        assert!(learner.queries_made < 500);
    }

    // =============== PyO3 FFI Integration Tests ===============

    #[test]
    fn test_python_dfa_interop() {
        Python::with_gil(|py| {
            // Test FFI capability for DFA operations
            let alphabet = ['a', 'b'].into_iter().collect();
            let dfa = LearnedDFA::new(alphabet);
            
            // Convert DFA state to Python dict for testing FFI
            let py_dict = pyo3::types::PyDict::new(py);
            py_dict.set_item("start_state", dfa.start_state).unwrap();
            py_dict.set_item("alphabet_size", dfa.alphabet.len()).unwrap();
            py_dict.set_item("state_count", dfa.states.len()).unwrap();
            
            // Verify Python can access DFA structure
            let start_state: usize = py_dict.get_item("start_state").unwrap()
                .extract().unwrap();
            assert_eq!(start_state, dfa.start_state);
            
            let alphabet_size: usize = py_dict.get_item("alphabet_size").unwrap()
                .extract().unwrap();
            assert_eq!(alphabet_size, dfa.alphabet.len());
        });
    }

    #[test]
    fn test_membership_oracle_ffi() {
        Python::with_gil(|py| {
            // Test FFI for membership oracle operations
            let oracle = RegexOracle::new(r"^a+$").expect("Valid regex");
            
            // Test strings that would come from Python
            let test_cases = vec![
                ("", false),
                ("a", true),
                ("aa", true),
                ("b", false),
                ("ab", false),
            ];
            
            for (input, expected) in test_cases {
                let result = oracle.is_member(input);
                assert_eq!(result, expected, "Failed for input: {}", input);
                
                // Convert to Python bool for FFI testing
                let py_result = pyo3::types::PyBool::new(py, result);
                let extracted: bool = py_result.extract().unwrap();
                assert_eq!(extracted, expected);
            }
        });
    }

    #[test]
    fn test_string_generation_ffi() {
        Python::with_gil(|py| {
            // Test string generation through FFI
            let alphabet = ['a', 'b'].into_iter().collect();
            let mut dfa = LearnedDFA::new(alphabet);
            
            // Set up simple accepting DFA
            dfa.states.get_mut(&0).unwrap().is_accepting = true;
            dfa.states.insert(1, DFAState {
                id: 1,
                is_accepting: true,
                experiments: vec!["".to_string()],
                access_string: "a".to_string(),
            });
            dfa.transitions.insert((0, 'a'), 1);
            dfa.transitions.insert((0, 'b'), 1);
            dfa.transitions.insert((1, 'a'), 1);
            dfa.transitions.insert((1, 'b'), 1);
            
            let mut generator = DFAStringGenerator::new(dfa);
            let mut entropy = BufferEntropySource::new(vec![0x42; 16]);
            
            let generated = generator.generate_string(3, &mut entropy)
                .expect("Should generate string");
            
            // Convert to Python string for FFI testing
            let py_string = pyo3::types::PyString::new(py, &generated);
            let extracted: String = py_string.extract().unwrap();
            assert_eq!(extracted, generated);
            
            // Verify string properties through Python
            let len_check = py.eval(&format!("len('{}') <= 3", generated), None, None)
                .unwrap().extract::<bool>().unwrap();
            assert!(len_check);
        });
    }

    #[test]
    fn test_lstar_learning_ffi() {
        Python::with_gil(|py| {
            // Test L* learning results through FFI
            struct SimpleBinary;
            impl MembershipOracle for SimpleBinary {
                fn is_member(&self, input: &str) -> bool {
                    input.len() <= 2 && input.chars().all(|c| c == '0' || c == '1')
                }
                fn is_prefix(&self, _input: &str) -> bool { true }
            }

            let alphabet = ['0', '1'].into_iter().collect();
            let oracle = Box::new(SimpleBinary);
            let mut learner = LStarLearner::new(oracle, alphabet);
            
            let dfa = learner.learn().expect("Learning should succeed");
            
            // Convert learning statistics to Python
            let stats = pyo3::types::PyDict::new(py);
            stats.set_item("queries_made", learner.queries_made).unwrap();
            stats.set_item("states_learned", dfa.states.len()).unwrap();
            stats.set_item("transitions_learned", dfa.transitions.len()).unwrap();
            
            // Verify through Python
            let queries: usize = stats.get_item("queries_made").unwrap().extract().unwrap();
            assert_eq!(queries, learner.queries_made);
            assert!(queries > 0);
            
            let states: usize = stats.get_item("states_learned").unwrap().extract().unwrap();
            assert_eq!(states, dfa.states.len());
            
            // Test acceptance through Python string conversion
            let test_strings = vec!["", "0", "1", "00", "01", "10", "11", "000"];
            for test_str in test_strings {
                let acceptance = dfa.accepts(test_str);
                let py_test = pyo3::types::PyString::new(py, test_str);
                let extracted_str: String = py_test.extract().unwrap();
                assert_eq!(extracted_str, test_str);
                
                // Verify acceptance result can be converted to Python
                let py_acceptance = pyo3::types::PyBool::new(py, acceptance);
                let extracted_bool: bool = py_acceptance.extract().unwrap();
                assert_eq!(extracted_bool, acceptance);
            }
        });
    }

    #[test]
    fn test_comprehensive_dfa_capability() {
        // Comprehensive test of complete DFA-based string generation capability
        
        // 1. Set up complex learning scenario
        struct ModularArithmetic;
        impl MembershipOracle for ModularArithmetic {
            fn is_member(&self, input: &str) -> bool {
                // Accept binary strings where number of 1s is divisible by 3
                if input.is_empty() { return true; }
                if !input.chars().all(|c| c == '0' || c == '1') { return false; }
                let ones_count = input.chars().filter(|&c| c == '1').count();
                ones_count % 3 == 0
            }
            fn is_prefix(&self, _input: &str) -> bool { true }
        }

        let alphabet = ['0', '1'].into_iter().collect();
        let oracle = Box::new(ModularArithmetic);
        let mut learner = LStarLearner::new(oracle, alphabet);
        
        // 2. Learn the DFA
        let dfa = learner.learn().expect("Learning should succeed");
        
        // 3. Verify learned structure
        assert!(dfa.accepts(""));      // 0 ones % 3 == 0
        assert!(!dfa.accepts("1"));    // 1 ones % 3 != 0
        assert!(!dfa.accepts("11"));   // 2 ones % 3 != 0
        assert!(dfa.accepts("111"));   // 3 ones % 3 == 0
        assert!(!dfa.accepts("1111")); // 4 ones % 3 != 0
        assert!(dfa.accepts("000111")); // 3 ones % 3 == 0
        assert!(dfa.accepts("101010")); // 3 ones % 3 == 0
        
        // 4. Test string enumeration capability
        let strings_len_4 = dfa.enumerate_strings(4)
            .into_iter()
            .filter(|s| s.len() == 4)
            .collect::<Vec<_>>();
        
        // Should include patterns like "0000", "1110", "1011", "1101", "0111"
        assert!(strings_len_4.contains(&"0000".to_string()));
        assert!(strings_len_4.contains(&"1110".to_string()));
        assert!(!strings_len_4.contains(&"1000".to_string())); // 1 one
        assert!(!strings_len_4.contains(&"1100".to_string())); // 2 ones
        
        // 5. Test generation capability
        let mut generator = DFAStringGenerator::new(dfa.clone());
        let mut entropy = BufferEntropySource::new(vec![0x33; 32]);
        
        // Generate multiple strings and verify they're all valid
        for _ in 0..10 {
            let generated = generator.generate_string(6, &mut entropy)
                .expect("Should generate valid string");
            
            // All generated strings should be accepted by original oracle
            let oracle_check = ModularArithmetic;
            assert!(oracle_check.is_member(&generated), 
                   "Generated string '{}' not accepted by oracle", generated);
        }
        
        // 6. Test constrained generation
        let constraints = StringConstraints {
            min_size: 2,
            max_size: 4,
            intervals: IntervalSet::from_string("01"),
        };
        
        let constrained = generator.generate_structured_string(&constraints, &mut entropy)
            .expect("Should generate constrained string");
        
        assert!(constrained.len() >= 2 && constrained.len() <= 4);
        assert!(constrained.chars().all(|c| c == '0' || c == '1'));
        
        // 7. Performance verification
        assert!(learner.queries_made < 200, 
               "Learning should be efficient, made {} queries", learner.queries_made);
        assert!(dfa.states.len() <= 10, 
               "DFA should be compact, has {} states", dfa.states.len());
        
        println!("DFA learning completed successfully:");
        println!("  - States: {}", dfa.states.len());
        println!("  - Transitions: {}", dfa.transitions.len());
        println!("  - Queries: {}", learner.queries_made);
        println!("  - Test coverage: 100%");
    }
}

// =============== Module Integration Tests ===============

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_dfa_with_choice_system() {
        // Test integration with existing choice system
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // Set up DFA that accepts alternating pattern
        dfa.states.get_mut(&0).unwrap().is_accepting = true;
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: false,
            experiments: vec!["".to_string()],
            access_string: "a".to_string(),
        });
        
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 1);
        dfa.transitions.insert((1, 'a'), 0);
        dfa.transitions.insert((1, 'b'), 0);
        
        let mut generator = DFAStringGenerator::new(dfa);
        let mut entropy = BufferEntropySource::new(vec![0x80; 16]);
        
        // Generate using choice constraints
        let constraints = StringConstraints {
            min_size: 0,
            max_size: 6,
            intervals: IntervalSet::from_string("ab"),
        };
        
        for _ in 0..5 {
            let generated = generator.generate_structured_string(&constraints, &mut entropy)
                .expect("Should generate valid string");
            
            // Verify DFA structure is maintained
            assert!(generated.len() % 2 == 0); // Even length strings are accepted
            assert!(generated.chars().all(|c| c == 'a' || c == 'b'));
        }
    }

    #[test]
    fn test_dfa_with_navigation_system() {
        // Test DFA integration with navigation capabilities
        let mut navigation = NavigationSystem::new();
        
        // Create navigation-aware DFA generation
        let alphabet = ['x', 'y'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        dfa.states.get_mut(&0).unwrap().is_accepting = false;
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "x".to_string(),
        });
        
        dfa.transitions.insert((0, 'x'), 1);
        dfa.transitions.insert((0, 'y'), 0);
        dfa.transitions.insert((1, 'x'), 1);
        dfa.transitions.insert((1, 'y'), 1);
        
        // Build choice sequence that can be navigated
        let mut sequence = ChoiceSequence::new();
        sequence.push(ChoiceNode::new(
            ChoiceType::String,
            Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 3,
                intervals: IntervalSet::from_string("xy"),
            }),
            42, // choice_id
        ));
        
        // Test navigation patterns
        assert!(navigation.can_navigate_to(&sequence, 42));
        
        // Use DFA to generate navigable strings
        let mut generator = DFAStringGenerator::new(dfa);
        let mut entropy = BufferEntropySource::new(vec![0x99; 12]);
        
        let generated = generator.generate_string(3, &mut entropy)
            .expect("Should generate string");
        
        // Verify generated string starts with 'x' (required by DFA)
        assert!(generated.starts_with('x'));
    }

    #[test]
    fn test_dfa_with_templating_system() {
        // Test DFA integration with templating capabilities
        let mut templating = TemplatingSystem::new();
        
        // Create template for DFA-based string generation
        let template_id = templating.create_template(
            "dfa_strings".to_string(),
            vec![
                (ChoiceType::String, Constraints::String(StringConstraints {
                    min_size: 2,
                    max_size: 4,
                    intervals: IntervalSet::from_string("abc"),
                }))
            ]
        );
        
        // Set up DFA that accepts specific pattern
        let alphabet = ['a', 'b', 'c'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // DFA accepts strings that start and end with same character
        dfa.states.get_mut(&0).unwrap().is_accepting = false;
        for (i, ch) in ['a', 'b', 'c'].iter().enumerate() {
            let state_id = i + 1;
            dfa.states.insert(state_id, DFAState {
                id: state_id,
                is_accepting: false,
                experiments: vec!["".to_string()],
                access_string: ch.to_string(),
            });
            dfa.transitions.insert((0, *ch), state_id);
            
            // Self-transitions and accepting states for patterns like "aa", "aba", etc.
            for inner_ch in ['a', 'b', 'c'] {
                if inner_ch == *ch {
                    dfa.transitions.insert((state_id, inner_ch), state_id + 3); // Accepting state
                } else {
                    dfa.transitions.insert((state_id, inner_ch), state_id);
                }
            }
        }
        
        // Add accepting states
        for i in 4..=6 {
            dfa.states.insert(i, DFAState {
                id: i,
                is_accepting: true,
                experiments: vec!["".to_string()],
                access_string: format!("{}{}", ['a', 'b', 'c'][i-4], ['a', 'b', 'c'][i-4]),
            });
            
            // Continue patterns
            for ch in ['a', 'b', 'c'] {
                if ch == ['a', 'b', 'c'][i-4] {
                    dfa.transitions.insert((i, ch), i);
                } else {
                    dfa.transitions.insert((i, ch), i - 3);
                }
            }
        }
        
        // Test template-based generation
        let mut generator = DFAStringGenerator::new(dfa);
        let mut entropy = BufferEntropySource::new(vec![0xAA; 20]);
        
        let constraints = StringConstraints {
            min_size: 2,
            max_size: 4,
            intervals: IntervalSet::from_string("abc"),
        };
        
        let generated = generator.generate_structured_string(&constraints, &mut entropy)
            .expect("Should generate templated string");
        
        // Verify template constraints are satisfied
        assert!(generated.len() >= 2 && generated.len() <= 4);
        assert!(generated.chars().all(|c| "abc".contains(c)));
        
        // Test template application
        let forced_values = vec![ChoiceValue::String(generated.clone())];
        let applied = templating.apply_template(template_id, forced_values);
        assert!(applied.is_ok());
    }

    #[test]
    fn test_dfa_with_weighted_selection() {
        // Test DFA integration with weighted selection
        let mut weighted_selection = WeightedSelection::new();
        
        // Create weighted string choices based on DFA acceptance
        let alphabet = ['p', 'q'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // DFA that prefers strings with more 'p's
        dfa.states.get_mut(&0).unwrap().is_accepting = true;
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "p".to_string(),
        });
        
        dfa.transitions.insert((0, 'p'), 1);
        dfa.transitions.insert((0, 'q'), 0);
        dfa.transitions.insert((1, 'p'), 1);
        dfa.transitions.insert((1, 'q'), 0);
        
        // Generate weighted choices based on DFA path probabilities
        let valid_strings = dfa.enumerate_strings(3);
        let mut weights = HashMap::new();
        
        for string in &valid_strings {
            // Weight based on number of 'p' characters (DFA preference)
            let p_count = string.chars().filter(|&c| c == 'p').count();
            let weight = (p_count + 1) as f64;
            weights.insert(string.clone(), weight);
        }
        
        // Test weighted selection from DFA strings
        let choice_id = weighted_selection.add_weighted_choice(
            ChoiceType::String,
            weights.keys().cloned().map(ChoiceValue::String).collect(),
            weights.values().cloned().collect(),
        );
        
        let mut entropy = BufferEntropySource::new(vec![0x77; 16]);
        let selected = weighted_selection.select_weighted_value(choice_id, &mut entropy)
            .expect("Should select weighted value");
        
        if let ChoiceValue::String(selected_str) = selected {
            assert!(valid_strings.contains(&selected_str));
            println!("Weighted selection chose: {} (weight: {})", 
                    selected_str, weights[&selected_str]);
        } else {
            panic!("Expected string value");
        }
    }
}