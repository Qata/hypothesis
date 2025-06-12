//! Basic test for DFA-Based String Generation System verification
//!
//! This simplified test verifies the core DFA functionality without
//! extensive integration testing.

use crate::choice::{ChoiceValue, ChoiceType, Constraints, StringConstraints};
use std::collections::{HashMap, HashSet, VecDeque};

/// Membership oracle that determines if strings belong to target language
pub trait MembershipOracle {
    fn is_member(&self, input: &str) -> bool;
    fn is_prefix(&self, input: &str) -> bool;
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

        results.sort();
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_dfa_creation() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let dfa = LearnedDFA::new(alphabet);
        
        assert_eq!(dfa.start_state, 0);
        assert!(dfa.states.contains_key(&0));
        assert_eq!(dfa.next_state_id, 1);
    }

    #[test]
    fn test_dfa_string_acceptance() {
        let alphabet = ['a', 'b'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // Set up a simple DFA that accepts strings ending with 'a'
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
        let alphabet = ['a'].into_iter().collect();
        let mut dfa = LearnedDFA::new(alphabet);
        
        // Set up a DFA that accepts "a" and "aa"
        dfa.states.insert(1, DFAState {
            id: 1,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "a".to_string(),
        });
        dfa.states.insert(2, DFAState {
            id: 2,
            is_accepting: true,
            experiments: vec!["".to_string()],
            access_string: "aa".to_string(),
        });
        
        dfa.transitions.insert((0, 'a'), 1);
        dfa.transitions.insert((1, 'a'), 2);
        
        let strings = dfa.enumerate_strings(2);
        assert_eq!(strings, vec!["a", "aa"]);
    }

    #[test]
    fn test_membership_oracle_trait() {
        struct ContainsA;
        impl MembershipOracle for ContainsA {
            fn is_member(&self, input: &str) -> bool {
                input.contains('a')
            }
            fn is_prefix(&self, input: &str) -> bool {
                self.is_member(input) || input.is_empty()
            }
        }

        let oracle = ContainsA;
        assert!(oracle.is_member("a"));
        assert!(oracle.is_member("ba"));
        assert!(!oracle.is_member("b"));
        assert!(oracle.is_prefix(""));
    }
}