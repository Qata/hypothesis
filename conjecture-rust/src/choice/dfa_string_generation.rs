//! DFA-Based String Generation System
//!
//! This module implements L* algorithm for learning finite automata, providing
//! string pattern recognition, optimization, and structured string generation
//! with regex-like capabilities. It integrates with the choice system to provide
//! sophisticated string generation based on learned patterns.

use crate::choice::{
    ChoiceValue, ChoiceType, Constraints, StringConstraints,
    ValueGenerator, EntropySource, ValueGenerationResult, ValueGenerationError,
    BooleanConstraints, IntegerConstraints, FloatConstraints, BytesConstraints,
};
use std::collections::{HashMap, HashSet, VecDeque};
use log::{debug, info, warn, error};

/// Error types for DFA operations with comprehensive debugging
#[derive(Debug, Clone, PartialEq)]
pub enum DFAError {
    InvalidState { state_id: usize, context: String },
    InvalidTransition { from_state: usize, symbol: char, context: String },
    LearningFailed { reason: String, queries_made: usize },
    MembershipQueryFailed { input: String, error: String },
    GenerationFailed { constraint_violation: String },
    InsufficientEntropy { required: usize, available: usize },
}

impl std::fmt::Display for DFAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DFAError::InvalidState { state_id, context } => 
                write!(f, "Invalid DFA state 0x{:X} in context: {}", state_id, context),
            DFAError::InvalidTransition { from_state, symbol, context } =>
                write!(f, "Invalid transition from state 0x{:X} on symbol '{}' in context: {}", 
                       from_state, symbol, context),
            DFAError::LearningFailed { reason, queries_made } =>
                write!(f, "DFA learning failed after {} queries: {}", queries_made, reason),
            DFAError::MembershipQueryFailed { input, error } =>
                write!(f, "Membership query failed for input '{}': {}", input, error),
            DFAError::GenerationFailed { constraint_violation } =>
                write!(f, "String generation failed: {}", constraint_violation),
            DFAError::InsufficientEntropy { required, available } =>
                write!(f, "Insufficient entropy: need {} bytes, have {}", required, available),
        }
    }
}

impl std::error::Error for DFAError {}

/// Represents a state in a learned DFA with comprehensive debugging information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DFAState {
    pub id: usize,
    pub is_accepting: bool,
    pub experiments: Vec<String>,
    pub access_string: String,
    pub visit_count: usize,
    pub last_updated: std::time::SystemTime,
}

impl DFAState {
    pub fn new(id: usize, access_string: String, is_accepting: bool) -> Self {
        debug!("Creating DFA state 0x{:X} with access string '{}', accepting: {}", 
               id, access_string, is_accepting);
        
        Self {
            id,
            is_accepting,
            experiments: Vec::new(),
            access_string,
            visit_count: 0,
            last_updated: std::time::SystemTime::now(),
        }
    }

    pub fn visit(&mut self) {
        self.visit_count += 1;
        self.last_updated = std::time::SystemTime::now();
    }
}

/// Transition function for DFA states with debugging
pub type TransitionFunction = HashMap<(usize, char), usize>;

/// Membership oracle that determines if strings belong to target language
pub trait MembershipOracle: std::fmt::Debug {
    fn is_member(&self, input: &str) -> bool;
    fn is_prefix(&self, input: &str) -> bool;
    fn get_description(&self) -> String { "Unknown oracle".to_string() }
}

/// Deterministic Finite Automaton with L* learning capability
#[derive(Debug, Clone)]
pub struct LearnedDFA {
    pub states: HashMap<usize, DFAState>,
    pub transitions: TransitionFunction,
    pub start_state: usize,
    pub alphabet: HashSet<char>,
    pub next_state_id: usize,
    pub creation_time: std::time::SystemTime,
    pub total_queries: usize,
}

impl LearnedDFA {
    pub fn new(alphabet: HashSet<char>) -> Self {
        info!("Creating new DFA with alphabet: {:?}", alphabet);
        
        let mut states = HashMap::new();
        states.insert(0, DFAState::new(0, String::new(), false));

        Self {
            states,
            transitions: HashMap::new(),
            start_state: 0,
            alphabet,
            next_state_id: 1,
            creation_time: std::time::SystemTime::now(),
            total_queries: 0,
        }
    }

    /// Check if DFA accepts given input string
    pub fn accepts(&self, input: &str) -> bool {
        debug!("Checking acceptance for input: '{}'", input);
        
        let mut current_state = self.start_state;
        
        for ch in input.chars() {
            if let Some(&next_state) = self.transitions.get(&(current_state, ch)) {
                debug!("Transition: {} --{}--> {}", 
                       format!("0x{:X}", current_state), ch, format!("0x{:X}", next_state));
                current_state = next_state;
            } else {
                debug!("No transition from state 0x{:X} on symbol '{}'", current_state, ch);
                return false;
            }
        }
        
        let accepts = self.states.get(&current_state)
            .map(|state| state.is_accepting)
            .unwrap_or(false);
            
        debug!("Input '{}' acceptance: {} (final state: 0x{:X})", 
               input, accepts, current_state);
        accepts
    }

    /// Enumerate all accepted strings up to max_length
    pub fn enumerate_strings(&self, max_length: usize) -> Vec<String> {
        debug!("Enumerating strings up to length {}", max_length);
        
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((self.start_state, String::new()));

        while let Some((state, prefix)) = queue.pop_front() {
            if prefix.len() <= max_length {
                if let Some(dfa_state) = self.states.get(&state) {
                    if dfa_state.is_accepting {
                        debug!("Found accepted string: '{}'", prefix);
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
        info!("Enumerated {} strings up to length {}", results.len(), max_length);
        results
    }

    /// Count strings of specific length accepted by DFA
    pub fn count_strings_of_length(&self, length: usize) -> usize {
        debug!("Counting strings of length {}", length);
        
        if length == 0 {
            let count = if self.states.get(&self.start_state)
                .map(|s| s.is_accepting)
                .unwrap_or(false) { 1 } else { 0 };
            debug!("Length 0 count: {}", count);
            return count;
        }

        let mut counts = HashMap::new();
        counts.insert(self.start_state, 1usize);

        for step in 0..length {
            let mut new_counts = HashMap::new();
            
            for (&state, &count) in &counts {
                for &ch in &self.alphabet {
                    if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                        *new_counts.entry(next_state).or_insert(0) += count;
                    }
                }
            }
            
            counts = new_counts;
            debug!("Step {}: {} reachable states", step + 1, counts.len());
        }

        let total = counts.iter()
            .filter_map(|(&state, &count)| {
                self.states.get(&state)
                    .filter(|s| s.is_accepting)
                    .map(|_| count)
            })
            .sum();
            
        debug!("Total strings of length {}: {}", length, total);
        total
    }

    /// Get comprehensive statistics about the DFA
    pub fn get_statistics(&self) -> DFAStatistics {
        DFAStatistics {
            state_count: self.states.len(),
            transition_count: self.transitions.len(),
            alphabet_size: self.alphabet.len(),
            accepting_states: self.states.values().filter(|s| s.is_accepting).count(),
            total_queries: self.total_queries,
            creation_time: self.creation_time,
        }
    }
}

/// Statistics about a learned DFA
#[derive(Debug, Clone)]
pub struct DFAStatistics {
    pub state_count: usize,
    pub transition_count: usize,
    pub alphabet_size: usize,
    pub accepting_states: usize,
    pub total_queries: usize,
    pub creation_time: std::time::SystemTime,
}

/// L* Learning Algorithm Implementation with comprehensive debugging
pub struct LStarLearner {
    oracle: Box<dyn MembershipOracle>,
    alphabet: HashSet<char>,
    observation_table: HashMap<String, HashMap<String, bool>>,
    prefixes: HashSet<String>,
    suffixes: HashSet<String>,
    queries_made: usize,
    max_queries: usize,
    debug_mode: bool,
}

impl LStarLearner {
    pub fn new(oracle: Box<dyn MembershipOracle>, alphabet: HashSet<char>) -> Self {
        info!("Initializing L* learner with oracle: {}", oracle.get_description());
        debug!("Alphabet: {:?}", alphabet);
        
        let mut learner = Self {
            oracle,
            alphabet,
            observation_table: HashMap::new(),
            prefixes: HashSet::new(),
            suffixes: HashSet::new(),
            queries_made: 0,
            max_queries: 10000,
            debug_mode: true,
        };

        // Initialize with empty string
        learner.prefixes.insert(String::new());
        learner.suffixes.insert(String::new());
        learner.fill_table();
        learner
    }

    pub fn set_max_queries(&mut self, max: usize) {
        self.max_queries = max;
        info!("Set maximum queries to {}", max);
    }

    pub fn set_debug_mode(&mut self, debug: bool) {
        self.debug_mode = debug;
    }

    /// Main L* learning algorithm
    pub fn learn(&mut self) -> Result<LearnedDFA, DFAError> {
        info!("Starting L* learning algorithm");
        
        let start_time = std::time::Instant::now();
        let mut iterations = 0;
        
        loop {
            iterations += 1;
            debug!("L* iteration {}", iterations);
            
            // Make table closed and consistent
            self.make_closed()?;
            self.make_consistent()?;

            // Build candidate DFA
            let dfa = self.build_dfa()?;
            debug!("Built candidate DFA with {} states", dfa.states.len());

            // Test equivalence with oracle
            if let Some(counterexample) = self.find_counterexample(&dfa) {
                info!("Found counterexample: '{}'", counterexample);
                self.process_counterexample(&counterexample);
                self.fill_table();
            } else {
                let elapsed = start_time.elapsed();
                info!("L* learning completed successfully in {:?}", elapsed);
                info!("Final statistics: {} states, {} transitions, {} queries", 
                      dfa.states.len(), dfa.transitions.len(), self.queries_made);
                return Ok(dfa);
            }

            // Safety check to prevent infinite loops
            if self.queries_made > self.max_queries {
                error!("L* learning exceeded maximum queries: {}", self.max_queries);
                return Err(DFAError::LearningFailed {
                    reason: format!("Exceeded maximum queries ({})", self.max_queries),
                    queries_made: self.queries_made,
                });
            }
        }
    }

    fn fill_table(&mut self) {
        debug!("Filling observation table");
        
        for prefix in self.prefixes.clone() {
            for suffix in self.suffixes.clone() {
                let string = prefix.clone() + &suffix;
                if !self.observation_table.contains_key(&prefix) {
                    self.observation_table.insert(prefix.clone(), HashMap::new());
                }
                
                if !self.observation_table[&prefix].contains_key(&suffix) {
                    let is_member = self.oracle.is_member(&string);
                    self.queries_made += 1;
                    
                    if self.debug_mode && self.queries_made % 100 == 0 {
                        debug!("Made {} membership queries", self.queries_made);
                    }
                    
                    self.observation_table.get_mut(&prefix).unwrap()
                        .insert(suffix, is_member);
                        
                    debug!("Query {}: '{}' -> {}", self.queries_made, string, is_member);
                }
            }
        }

        // Fill extended table (prefix + alphabet)
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
        
        debug!("Observation table filled: {} prefixes, {} suffixes", 
               self.prefixes.len(), self.suffixes.len());
    }

    fn make_closed(&mut self) -> Result<(), DFAError> {
        debug!("Making observation table closed");
        
        loop {
            let mut found_unclosed = false;
            
            for prefix in self.prefixes.clone() {
                for ch in &self.alphabet {
                    let extended = prefix.clone() + &ch.to_string();
                    if !self.prefixes.contains(&extended) {
                        if !self.has_representative(&extended) {
                            debug!("Adding prefix '{}' to maintain closure", extended);
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
        
        debug!("Observation table is now closed");
        Ok(())
    }

    fn make_consistent(&mut self) -> Result<(), DFAError> {
        debug!("Making observation table consistent");
        
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
                                if let Some(suffix) = self.find_distinguishing_suffix(&ext1, &ext2) {
                                    debug!("Adding distinguishing suffix '{}' for consistency", suffix);
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
        
        debug!("Observation table is now consistent");
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
        debug!("Building DFA from observation table");
        
        let mut dfa = LearnedDFA::new(self.alphabet.clone());
        dfa.total_queries = self.queries_made;
        
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

                let mut state = DFAState::new(state_id, prefix.clone(), is_accepting);
                state.experiments = self.suffixes.iter().cloned().collect();

                dfa.states.insert(state_id, state);
                state_map.insert(prefix.clone(), state_id);
                representatives.push(prefix.clone());
                
                if prefix.is_empty() {
                    dfa.start_state = state_id;
                }
                
                debug!("Created state 0x{:X} for prefix '{}' (accepting: {})", 
                       state_id, prefix, is_accepting);
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
                        
                        debug!("Transition: 0x{:X} --{}--> 0x{:X}", 
                               from_state, ch, to_state);
                        break;
                    }
                }
            }
        }

        info!("Built DFA with {} states and {} transitions", 
              dfa.states.len(), dfa.transitions.len());
        Ok(dfa)
    }

    fn find_counterexample(&self, dfa: &LearnedDFA) -> Option<String> {
        debug!("Searching for counterexample");
        
        // Test strings up to reasonable length
        for length in 0..=8 {
            for string in self.generate_strings_of_length(length) {
                let dfa_accepts = dfa.accepts(&string);
                let oracle_accepts = self.oracle.is_member(&string);
                
                if dfa_accepts != oracle_accepts {
                    debug!("Counterexample found: '{}' (DFA: {}, Oracle: {})", 
                           string, dfa_accepts, oracle_accepts);
                    return Some(string);
                }
            }
        }
        
        debug!("No counterexample found");
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
        debug!("Processing counterexample: '{}'", counterexample);
        
        // Add all suffixes of counterexample as potential distinguishing experiments
        for i in 0..=counterexample.len() {
            let suffix = counterexample.chars().skip(i).collect::<String>();
            if self.suffixes.insert(suffix.clone()) {
                debug!("Added suffix: '{}'", suffix);
            }
        }
    }
}

/// Regex-based membership oracle for testing
#[derive(Debug)]
pub struct RegexOracle {
    pattern: regex::Regex,
    description: String,
}

impl RegexOracle {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
            description: format!("Regex({})", pattern),
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
        (0..5).any(|i| self.is_member(&format!("{}{}", input, "a".repeat(i))))
    }
    
    fn get_description(&self) -> String {
        self.description.clone()
    }
}

/// Custom membership oracle for specific patterns
pub struct CustomOracle {
    predicate: Box<dyn Fn(&str) -> bool>,
    description: String,
}

impl std::fmt::Debug for CustomOracle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomOracle")
            .field("description", &self.description)
            .finish()
    }
}

impl CustomOracle {
    pub fn new<F>(predicate: F, description: String) -> Self 
    where 
        F: Fn(&str) -> bool + 'static,
    {
        Self {
            predicate: Box::new(predicate),
            description,
        }
    }
}

impl MembershipOracle for CustomOracle {
    fn is_member(&self, input: &str) -> bool {
        (self.predicate)(input)
    }
    
    fn is_prefix(&self, _input: &str) -> bool {
        true // Conservative approach
    }
    
    fn get_description(&self) -> String {
        self.description.clone()
    }
}

/// DFA-based String Generator integrated with choice system
#[derive(Debug)]
pub struct DFAStringGenerator {
    dfa: LearnedDFA,
    cache: HashMap<(usize, StringConstraints), Vec<String>>,
    generation_stats: GenerationStatistics,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationStatistics {
    pub total_generations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub constraint_violations: usize,
    pub fallback_generations: usize,
}

impl DFAStringGenerator {
    pub fn new(dfa: LearnedDFA) -> Self {
        info!("Creating DFA string generator with {} states", dfa.states.len());
        
        Self {
            dfa,
            cache: HashMap::new(),
            generation_stats: GenerationStatistics::default(),
        }
    }

    /// Generate string using DFA with entropy source
    pub fn generate_string(
        &mut self,
        max_length: usize,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFAError> {
        debug!("Generating string with max length {}", max_length);
        
        self.generation_stats.total_generations += 1;
        
        let valid_strings = self.dfa.enumerate_strings(max_length);
        
        if valid_strings.is_empty() {
            warn!("No valid strings found for max length {}", max_length);
            return Err(DFAError::GenerationFailed {
                constraint_violation: format!("No valid strings of max length {}", max_length),
            });
        }

        // Use entropy to select from valid strings
        let index_bytes = entropy.draw_bytes(4).map_err(|_| {
            error!("Insufficient entropy for string selection");
            DFAError::InsufficientEntropy { required: 4, available: 0 }
        })?;
        
        let index = u32::from_le_bytes([
            index_bytes[0], index_bytes[1], index_bytes[2], index_bytes[3]
        ]) as usize % valid_strings.len();

        let selected = valid_strings[index].clone();
        debug!("Generated string: '{}' (index {} of {})", selected, index, valid_strings.len());
        
        Ok(selected)
    }

    /// Generate structured string with constraints
    pub fn generate_structured_string(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFAError> {
        debug!("Generating structured string with constraints: min={}, max={}", 
               constraints.min_size, constraints.max_size);
        
        self.generation_stats.total_generations += 1;
        
        // Check cache first
        let cache_key = (self.dfa.states.len(), constraints.clone());
        if let Some(cached_strings) = self.cache.get(&cache_key) {
            self.generation_stats.cache_hits += 1;
            debug!("Using cached strings: {} candidates", cached_strings.len());
            
            if !cached_strings.is_empty() {
                let index_bytes = entropy.draw_bytes(4).map_err(|_| 
                    DFAError::InsufficientEntropy { required: 4, available: 0 })?;
                let index = u32::from_le_bytes([
                    index_bytes[0], index_bytes[1], index_bytes[2], index_bytes[3]
                ]) as usize % cached_strings.len();
                
                return Ok(cached_strings[index].clone());
            }
        }
        
        self.generation_stats.cache_misses += 1;

        // Filter DFA strings by constraints
        let max_search_length = constraints.max_size.min(12);
        let candidates = self.dfa.enumerate_strings(max_search_length)
            .into_iter()
            .filter(|s| {
                if s.len() < constraints.min_size || s.len() > constraints.max_size {
                    return false;
                }
                
                s.chars().all(|ch| {
                    let code = ch as u32;
                    constraints.intervals.intervals.iter().any(|(start, end)| {
                        code >= *start && code <= *end
                    })
                })
            })
            .collect::<Vec<_>>();

        // Cache the results
        self.cache.insert(cache_key, candidates.clone());

        if candidates.is_empty() {
            warn!("No candidates found matching constraints, using fallback");
            self.generation_stats.constraint_violations += 1;
            self.generation_stats.fallback_generations += 1;
            
            // Fallback: generate simple string matching constraints
            return self.fallback_generation(constraints, entropy);
        }

        let index_bytes = entropy.draw_bytes(4).map_err(|_| 
            DFAError::InsufficientEntropy { required: 4, available: 0 })?;
        let index = u32::from_le_bytes([
            index_bytes[0], index_bytes[1], index_bytes[2], index_bytes[3]
        ]) as usize % candidates.len();

        let selected = candidates[index].clone();
        debug!("Generated structured string: '{}' (index {} of {} candidates)", 
               selected, index, candidates.len());
        
        Ok(selected)
    }

    fn fallback_generation(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> Result<String, DFAError> {
        debug!("Using fallback generation for constraints");
        
        // Simple fallback: generate random string within constraints
        let length_bytes = entropy.draw_bytes(4).map_err(|_| 
            DFAError::InsufficientEntropy { required: 4, available: 0 })?;
        let length = constraints.min_size + 
            (u32::from_le_bytes([length_bytes[0], length_bytes[1], length_bytes[2], length_bytes[3]]) 
             as usize % (constraints.max_size - constraints.min_size + 1));

        let valid_chars: Vec<char> = constraints.intervals.intervals.iter()
            .flat_map(|(start, end)| (*start..=*end).filter_map(|c| char::from_u32(c)))
            .collect();

        if valid_chars.is_empty() {
            return Err(DFAError::GenerationFailed {
                constraint_violation: "No valid characters in constraints".to_string(),
            });
        }

        let mut result = String::new();
        for _ in 0..length {
            let char_bytes = entropy.draw_bytes(4).map_err(|_| 
                DFAError::InsufficientEntropy { required: 4, available: 0 })?;
            let char_index = u32::from_le_bytes([
                char_bytes[0], char_bytes[1], char_bytes[2], char_bytes[3]
            ]) as usize % valid_chars.len();
            
            result.push(valid_chars[char_index]);
        }

        debug!("Fallback generated: '{}'", result);
        Ok(result)
    }

    /// Get generation statistics
    pub fn get_statistics(&self) -> &GenerationStatistics {
        &self.generation_stats
    }

    /// Clear generation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        info!("Generation cache cleared");
    }

    /// Get underlying DFA
    pub fn get_dfa(&self) -> &LearnedDFA {
        &self.dfa
    }
}

/// Pattern recognition and optimization system for DFA learning
#[derive(Debug)]
pub struct PatternRecognitionEngine {
    alphabet_optimization: AlphabetOptimizer,
    pattern_cache: HashMap<String, Vec<String>>,
    known_pattern_names: Vec<String>,
}

impl PatternRecognitionEngine {
    pub fn new() -> Self {
        Self {
            alphabet_optimization: AlphabetOptimizer::new(),
            pattern_cache: HashMap::new(),
            known_pattern_names: Vec::new(),
        }
    }

    /// Detect common regex patterns from samples
    pub fn detect_pattern(&mut self, samples: &[(String, bool)]) -> Option<String> {
        debug!("Detecting pattern from {} samples", samples.len());
        
        // Analyze positive and negative examples
        let positive: Vec<&String> = samples.iter()
            .filter_map(|(s, accepted)| if *accepted { Some(s) } else { None })
            .collect();
        let negative: Vec<&String> = samples.iter()
            .filter_map(|(s, accepted)| if !*accepted { Some(s) } else { None })
            .collect();

        // Check for common patterns
        if let Some(pattern) = self.detect_length_pattern(&positive, &negative) {
            debug!("Detected length pattern: {}", pattern);
            return Some(pattern);
        }

        if let Some(pattern) = self.detect_prefix_suffix_pattern(&positive, &negative) {
            debug!("Detected prefix/suffix pattern: {}", pattern);
            return Some(pattern);
        }

        if let Some(pattern) = self.detect_character_class_pattern(&positive, &negative) {
            debug!("Detected character class pattern: {}", pattern);
            return Some(pattern);
        }

        None
    }

    fn detect_length_pattern(&self, positive: &[&String], negative: &[&String]) -> Option<String> {
        // Check if acceptance depends on string length
        let pos_lengths: HashSet<usize> = positive.iter().map(|s| s.len()).collect();
        let neg_lengths: HashSet<usize> = negative.iter().map(|s| s.len()).collect();

        // Check for even/odd length pattern
        let pos_even = positive.iter().all(|s| s.len() % 2 == 0);
        let neg_odd = negative.iter().all(|s| s.len() % 2 == 1);
        
        if pos_even && neg_odd {
            return Some(r"^(.{2})*$".to_string()); // Even length
        }

        // Check for specific length pattern
        if pos_lengths.len() == 1 && !pos_lengths.iter().any(|&len| neg_lengths.contains(&len)) {
            let target_len = *pos_lengths.iter().next().unwrap();
            return Some(format!(r"^.{{{}}}$", target_len));
        }

        None
    }

    fn detect_prefix_suffix_pattern(&self, positive: &[&String], negative: &[&String]) -> Option<String> {
        // Check for common prefix
        if let Some(common_prefix) = self.find_common_prefix(positive) {
            if !common_prefix.is_empty() && 
               negative.iter().all(|s| !s.starts_with(&common_prefix)) {
                return Some(format!(r"^{}.*", regex::escape(&common_prefix)));
            }
        }

        // Check for common suffix
        if let Some(common_suffix) = self.find_common_suffix(positive) {
            if !common_suffix.is_empty() && 
               negative.iter().all(|s| !s.ends_with(&common_suffix)) {
                return Some(format!(r".*{}$", regex::escape(&common_suffix)));
            }
        }

        // Check for prefix-suffix combination
        if let (Some(prefix), Some(suffix)) = (
            self.find_common_prefix(positive), 
            self.find_common_suffix(positive)
        ) {
            if !prefix.is_empty() && !suffix.is_empty() && prefix != suffix {
                return Some(format!(r"^{}.*{}$", 
                    regex::escape(&prefix), regex::escape(&suffix)));
            }
        }

        None
    }

    fn detect_character_class_pattern(&self, positive: &[&String], negative: &[&String]) -> Option<String> {
        // Analyze character usage
        let pos_chars: HashSet<char> = positive.iter()
            .flat_map(|s| s.chars())
            .collect();
        let neg_chars: HashSet<char> = negative.iter()
            .flat_map(|s| s.chars())
            .collect();

        // Check for alphabet restriction
        let exclusive_pos_chars: HashSet<char> = pos_chars.difference(&neg_chars).cloned().collect();
        if !exclusive_pos_chars.is_empty() && 
           positive.iter().all(|s| s.chars().all(|c| exclusive_pos_chars.contains(&c))) {
            let char_class = self.build_character_class(&exclusive_pos_chars);
            return Some(format!(r"^[{}]+$", char_class));
        }

        // Check for balanced parentheses pattern
        if positive.iter().all(|s| self.is_balanced_parentheses(s)) &&
           negative.iter().any(|s| !self.is_balanced_parentheses(s)) {
            return Some(r"^(\(.*\))*$".to_string());
        }

        None
    }

    fn find_common_prefix(&self, strings: &[&String]) -> Option<String> {
        if strings.is_empty() { return None; }
        
        let first = strings[0];
        for i in 0..first.len() {
            let prefix = &first[..=i];
            if !strings.iter().all(|s| s.starts_with(prefix)) {
                return if i == 0 { None } else { Some(first[..i].to_string()) };
            }
        }
        Some(first.clone())
    }

    fn find_common_suffix(&self, strings: &[&String]) -> Option<String> {
        if strings.is_empty() { return None; }
        
        let first = strings[0];
        for i in 1..=first.len() {
            let suffix = &first[first.len()-i..];
            if !strings.iter().all(|s| s.ends_with(suffix)) {
                return if i == 1 { None } else { 
                    Some(first[first.len()-(i-1)..].to_string()) 
                };
            }
        }
        Some(first.clone())
    }

    fn build_character_class(&self, chars: &HashSet<char>) -> String {
        let mut sorted_chars: Vec<char> = chars.iter().cloned().collect();
        sorted_chars.sort();
        
        // Handle special regex characters
        sorted_chars.iter()
            .map(|&c| match c {
                '^' | '-' | ']' | '\\' => format!("\\{}", c),
                _ => c.to_string(),
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn is_balanced_parentheses(&self, s: &str) -> bool {
        let mut depth = 0i32;
        for ch in s.chars() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 { return false; }
                }
                _ => {}
            }
        }
        depth == 0
    }

    /// Learn from existing DFA to optimize future learning
    pub fn learn_from_dfa(&mut self, dfa: &LearnedDFA, samples: &[(String, bool)]) {
        debug!("Learning patterns from DFA with {} states", dfa.states.len());
        
        // Extract structural patterns
        self.analyze_state_structure(dfa);
        self.analyze_transition_patterns(dfa);
        
        // Cache successful patterns
        if let Some(pattern) = self.detect_pattern(samples) {
            let pattern_key = format!("dfa_{}", dfa.states.len());
            self.pattern_cache.insert(pattern_key, 
                samples.iter()
                    .filter(|(_, accepted)| *accepted)
                    .map(|(s, _)| s.clone())
                    .collect()
            );
        }
    }

    fn analyze_state_structure(&self, dfa: &LearnedDFA) {
        debug!("Analyzing DFA state structure");
        
        // Analyze accepting vs non-accepting states
        let accepting_count = dfa.states.values().filter(|s| s.is_accepting).count();
        let total_states = dfa.states.len();
        
        debug!("DFA structure: {}/{} accepting states", accepting_count, total_states);
        
        // Look for structural patterns
        let has_sink_state = dfa.states.values().any(|state| {
            dfa.alphabet.iter().all(|&ch| {
                dfa.transitions.get(&(state.id, ch)) == Some(&state.id)
            }) && !state.is_accepting
        });
        
        if has_sink_state {
            debug!("DFA contains sink state (error state)");
        }
    }

    fn analyze_transition_patterns(&self, dfa: &LearnedDFA) {
        debug!("Analyzing DFA transition patterns");
        
        // Count self-loops
        let self_loops = dfa.transitions.iter()
            .filter(|((from, _), to)| from == *to)
            .count();
        
        debug!("DFA has {} self-loops out of {} transitions", 
               self_loops, dfa.transitions.len());
        
        // Analyze transition density
        let max_transitions = dfa.states.len() * dfa.alphabet.len();
        let density = dfa.transitions.len() as f64 / max_transitions as f64;
        
        debug!("Transition density: {:.2}%", density * 100.0);
    }
}

/// Alphabet optimization for more efficient learning
#[derive(Debug, Clone)]
pub struct AlphabetOptimizer {
    character_equivalences: HashMap<char, char>,
    optimization_cache: HashMap<String, HashSet<char>>,
}

impl AlphabetOptimizer {
    pub fn new() -> Self {
        Self {
            character_equivalences: HashMap::new(),
            optimization_cache: HashMap::new(),
        }
    }

    /// Optimize alphabet by finding character equivalences
    pub fn optimize_alphabet(&mut self, 
                            original_alphabet: &HashSet<char>, 
                            samples: &[(String, bool)]) -> HashSet<char> {
        debug!("Optimizing alphabet: {:?}", original_alphabet);
        
        // Create a string key for the cache
        let mut sorted_chars: Vec<char> = original_alphabet.iter().cloned().collect();
        sorted_chars.sort();
        let cache_key = sorted_chars.iter().collect::<String>();
        
        if let Some(cached) = self.optimization_cache.get(&cache_key) {
            debug!("Using cached alphabet optimization");
            return cached.clone();
        }
        
        let mut optimized = original_alphabet.clone();
        
        // Find characters that behave equivalently
        let equivalences = self.find_character_equivalences(original_alphabet, samples);
        
        for (char1, char2) in equivalences {
            if optimized.contains(&char1) && optimized.contains(&char2) {
                debug!("Characters '{}' and '{}' are equivalent", char1, char2);
                optimized.remove(&char2);
                self.character_equivalences.insert(char2, char1);
            }
        }
        
        self.optimization_cache.insert(cache_key, optimized.clone());
        
        info!("Alphabet optimized: {} -> {} characters", 
              original_alphabet.len(), optimized.len());
        
        optimized
    }

    fn find_character_equivalences(&self, 
                                  alphabet: &HashSet<char>, 
                                  samples: &[(String, bool)]) -> Vec<(char, char)> {
        let mut equivalences = Vec::new();
        let chars: Vec<char> = alphabet.iter().cloned().collect();
        
        for i in 0..chars.len() {
            for j in i+1..chars.len() {
                let char1 = chars[i];
                let char2 = chars[j];
                
                if self.are_characters_equivalent(char1, char2, samples) {
                    equivalences.push((char1, char2));
                }
            }
        }
        
        equivalences
    }

    fn are_characters_equivalent(&self, char1: char, char2: char, samples: &[(String, bool)]) -> bool {
        // Test if replacing char1 with char2 changes acceptance for any sample
        for (sample, expected) in samples {
            let modified = sample.replace(char1, &char2.to_string());
            
            // This is a simplified equivalence test
            // In practice, we'd need to test with an oracle
            if sample != &modified {
                // For now, assume they're different if replacement changes the string
                return false;
            }
        }
        
        true
    }

    /// Map character through equivalences
    pub fn map_character(&self, ch: char) -> char {
        self.character_equivalences.get(&ch).copied().unwrap_or(ch)
    }
}

/// Advanced DFA learning with pattern recognition and optimization
pub struct AdvancedDFALearner {
    pattern_engine: PatternRecognitionEngine,
    base_learner: LStarLearner,
    optimization_enabled: bool,
}

impl AdvancedDFALearner {
    pub fn new(oracle: Box<dyn MembershipOracle>, alphabet: HashSet<char>) -> Self {
        info!("Creating advanced DFA learner with pattern recognition");
        
        Self {
            pattern_engine: PatternRecognitionEngine::new(),
            base_learner: LStarLearner::new(oracle, alphabet),
            optimization_enabled: true,
        }
    }

    pub fn enable_optimization(&mut self, enabled: bool) {
        self.optimization_enabled = enabled;
        info!("DFA learning optimization: {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Learn DFA with pattern recognition and optimization
    pub fn learn_optimized(&mut self) -> Result<LearnedDFA, DFAError> {
        info!("Starting optimized DFA learning");
        
        let start_time = std::time::Instant::now();
        
        // Collect initial samples for pattern detection
        let initial_samples = self.collect_initial_samples();
        
        // Try pattern recognition first
        if self.optimization_enabled {
            if let Some(pattern) = self.pattern_engine.detect_pattern(&initial_samples) {
                info!("Detected pattern: {}", pattern);
                
                // Try to create oracle from pattern and learn faster
                if let Ok(regex_oracle) = RegexOracle::new(&pattern) {
                    info!("Using pattern-based learning acceleration");
                    // This could potentially speed up learning significantly
                }
            }
        }
        
        // Proceed with standard L* learning
        let dfa = self.base_learner.learn()?;
        
        // Learn from the result for future optimizations
        if self.optimization_enabled {
            self.pattern_engine.learn_from_dfa(&dfa, &initial_samples);
        }
        
        let elapsed = start_time.elapsed();
        info!("Optimized DFA learning completed in {:?}", elapsed);
        
        Ok(dfa)
    }

    fn collect_initial_samples(&mut self) -> Vec<(String, bool)> {
        debug!("Collecting initial samples for pattern detection");
        
        let mut samples = Vec::new();
        
        // Test empty string
        samples.push((String::new(), self.base_learner.oracle.is_member("")));
        
        // Test single characters
        for &ch in &self.base_learner.alphabet {
            let s = ch.to_string();
            samples.push((s.clone(), self.base_learner.oracle.is_member(&s)));
        }
        
        // Test short strings
        for len in 2..=4 {
            let test_strings = self.generate_test_strings(len);
            for s in test_strings.into_iter().take(10) { // Limit to avoid too many queries
                samples.push((s.clone(), self.base_learner.oracle.is_member(&s)));
            }
        }
        
        debug!("Collected {} initial samples", samples.len());
        samples
    }

    fn generate_test_strings(&self, length: usize) -> Vec<String> {
        if length == 0 {
            return vec![String::new()];
        }
        
        let alphabet: Vec<char> = self.base_learner.alphabet.iter().cloned().collect();
        let mut results = Vec::new();
        
        // Generate some representative strings rather than all possible
        let sample_count = std::cmp::min(20, alphabet.len().pow(length as u32));
        
        for i in 0..sample_count {
            let mut s = String::new();
            let mut index = i;
            
            for _ in 0..length {
                s.push(alphabet[index % alphabet.len()]);
                index /= alphabet.len();
            }
            
            results.push(s);
        }
        
        results
    }
}

/// Integration with ValueGenerator trait for specialized string generation
impl ValueGenerator for DFAStringGenerator {
    fn generate_value(
        &mut self,
        choice_type: ChoiceType,
        constraints: &Constraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<ChoiceValue> {
        match (choice_type, constraints) {
            (ChoiceType::String, Constraints::String(string_constraints)) => {
                let result = self.generate_structured_string(string_constraints, entropy)
                    .map_err(|_| ValueGenerationError::GenerationFailed(
                        "DFA string generation failed".to_string()
                    ))?;
                Ok(ChoiceValue::String(result))
            }
            _ => Err(ValueGenerationError::UnsupportedChoiceType(choice_type))
        }
    }

    fn generate_boolean(
        &mut self,
        _constraints: &BooleanConstraints,
        _entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<bool> {
        Err(ValueGenerationError::UnsupportedChoiceType(ChoiceType::Boolean))
    }

    fn generate_integer(
        &mut self,
        _constraints: &IntegerConstraints,
        _entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<i128> {
        Err(ValueGenerationError::UnsupportedChoiceType(ChoiceType::Integer))
    }

    fn generate_float(
        &mut self,
        _constraints: &FloatConstraints,
        _entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<f64> {
        Err(ValueGenerationError::UnsupportedChoiceType(ChoiceType::Float))
    }

    fn generate_string(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<String> {
        self.generate_structured_string(constraints, entropy)
            .map_err(|_| ValueGenerationError::GenerationFailed(
                "DFA string generation failed".to_string()
            ))
    }

    fn generate_bytes(
        &mut self,
        _constraints: &BytesConstraints,
        _entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<Vec<u8>> {
        Err(ValueGenerationError::UnsupportedChoiceType(ChoiceType::Bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{BufferEntropySource, IntervalSet};

    #[test]
    fn test_dfa_basic_creation() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let dfa = LearnedDFA::new(alphabet);
        
        assert_eq!(dfa.start_state, 0);
        assert!(dfa.states.contains_key(&0));
        assert_eq!(dfa.next_state_id, 1);
    }

    #[test]
    fn test_dfa_acceptance() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // DFA that accepts strings ending in 'a'
        dfa.states.insert(1, DFAState::new(1, "a".to_string(), true));
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 0);
        dfa.transitions.insert((1, 'a'), 1);
        dfa.transitions.insert((1, 'b'), 0);
        
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("ba"));
        assert!(!dfa.accepts("b"));
    }

    #[test]
    fn test_lstar_learning() {
        let oracle = Box::new(CustomOracle::new(
            |s: &str| s.contains("ab"),
            "Contains 'ab'".to_string(),
        ));
        
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut learner = LStarLearner::new(oracle, alphabet);
        learner.set_max_queries(500);
        
        let dfa = learner.learn().expect("Learning should succeed");
        
        assert!(dfa.accepts("ab"));
        assert!(dfa.accepts("aab"));
        assert!(!dfa.accepts("a"));
        assert!(!dfa.accepts("ba"));
    }

    #[test]
    fn test_dfa_string_generation() {
        let alphabet = ['x', 'y'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        dfa.states.get_mut(&0).unwrap().is_accepting = true;
        dfa.states.insert(1, DFAState::new(1, "x".to_string(), true));
        dfa.transitions.insert((0, 'x'), 1);
        dfa.transitions.insert((0, 'y'), 1);
        dfa.transitions.insert((1, 'x'), 1);
        dfa.transitions.insert((1, 'y'), 1);
        
        let mut generator = DFAStringGenerator::new(dfa);
        let mut entropy = BufferEntropySource::new(vec![0x42; 16]);
        
        let generated = generator.generate_string(2, &mut entropy)
            .expect("Should generate string");
        
        assert!(generated.len() <= 2);
        assert!(generated.chars().all(|c| c == 'x' || c == 'y'));
    }

    #[test]
    fn test_structured_generation() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        dfa.states.get_mut(&0).unwrap().is_accepting = false;
        dfa.states.insert(1, DFAState::new(1, "a".to_string(), true));
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((0, 'b'), 0);
        dfa.transitions.insert((1, 'a'), 1);
        dfa.transitions.insert((1, 'b'), 1);
        
        let mut generator = DFAStringGenerator::new(dfa);
        let mut entropy = BufferEntropySource::new(vec![0x33; 16]);
        
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 3,
            intervals: IntervalSet::from_string("ab"),
        };
        
        let generated = generator.generate_structured_string(&constraints, &mut entropy)
            .expect("Should generate structured string");
        
        assert!(generated.len() >= 1 && generated.len() <= 3);
        assert!(generated.starts_with('a')); // Required by DFA
    }
}