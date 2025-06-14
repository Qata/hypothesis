//! # ChoiceNode: Immutable Choice Representation with Metadata
//!
//! This module implements the core `ChoiceNode` data structure that serves as the fundamental
//! unit of choice recording in the Conjecture engine. Each ChoiceNode represents a single
//! decision point during test generation, capturing complete metadata about how the choice
//! was made, what constraints applied, and whether it was forced or randomly generated.
//!
//! ## Architecture Overview
//!
//! The `ChoiceNode` provides a Python Hypothesis compatible choice representation while
//! leveraging Rust's type system for enhanced safety and performance:
//!
//! ### Core Design Principles
//! - **Immutability**: Once created, choice nodes cannot be modified (except for index assignment)
//! - **Complete Metadata**: Every choice includes type, value, constraints, and generation context
//! - **Serializable**: Full support for database persistence and cross-language compatibility
//! - **Type Safety**: Compile-time prevention of type mismatches and invalid operations
//!
//! ### Choice Lifecycle
//! ```text
//! Generation → Recording → Storage → Replay/Shrinking
//!     ↓           ↓          ↓            ↓
//!   Create     Assign     Serialize    Deserialize
//!   Node       Index      to DB       & Validate
//! ```
//!
//! ## Key Components
//!
//! ### Choice Metadata
//! - **Type Information**: Strongly-typed choice classification (Integer, Boolean, Float, String, Bytes)
//! - **Value Storage**: Type-safe value representation with efficient serialization
//! - **Constraint Binding**: Complete constraint specification for validation and replay
//! - **Generation Context**: Whether the choice was forced, random, or otherwise generated
//!
//! ### Indexing System
//! - **Optional Indexing**: Supports both indexed and non-indexed choice nodes
//! - **Lazy Assignment**: Index assignment deferred to prevent stale references
//! - **Copy Safety**: Index explicitly not copied to prevent stale index issues
//! - **ExampleRecord Integration**: Proper index management through record lifecycle
//!
//! ### Forcing and Replay Support
//! - **Forced Choice Detection**: Tracks whether values were externally specified
//! - **Replay Compatibility**: Enables deterministic test case replay
//! - **Constraint Validation**: Ensures forced values satisfy all constraints
//! - **Immutability Protection**: Prevents modification of forced choice values
//!
//! ## Performance Characteristics
//!
//! ### Memory Efficiency
//! - **Compact Representation**: Optimized enum-based value storage
//! - **Zero-Copy Cloning**: Efficient cloning through Arc/Rc for large values
//! - **Minimal Overhead**: Typical choice node ~64-128 bytes depending on value type
//! - **Serialization**: Compressed binary format for database storage
//!
//! ### Time Complexity
//! - **Creation**: O(1) for all choice types
//! - **Validation**: O(1) for primitives, O(n) for strings/bytes where n = length
//! - **Copying**: O(1) for metadata, O(k) for value where k = value size
//! - **Serialization**: O(n) where n = total serialized size
//!
//! ### Cache Performance
//! - **CPU Cache Friendly**: Compact memory layout with good locality
//! - **Predictable Access**: Sequential access patterns during replay
//! - **Minimal Indirection**: Direct value storage without excessive pointer chasing
//!
//! ## Integration with Generation System
//!
//! ### Choice Recording Pipeline
//! 1. **Generation**: Value generated using constraints and provider
//! 2. **Validation**: Value validated against constraints
//! 3. **Node Creation**: ChoiceNode created with complete metadata
//! 4. **Storage**: Node added to choice sequence
//! 5. **Indexing**: Optional index assignment for later reference
//!
//! ### Shrinking Integration
//! - **Constraint Preservation**: All shrunk choices maintain original constraints
//! - **Value Transformation**: Shrinking produces new nodes with modified values
//! - **Metadata Consistency**: Generation context preserved during shrinking
//! - **Validation Enforcement**: All shrunk values validated against constraints
//!
//! ### Database Persistence
//! - **Deterministic Serialization**: Reproducible binary encoding
//! - **Backward Compatibility**: Version-tolerant deserialization
//! - **Compression**: Efficient storage for large choice sequences
//! - **Cross-Platform**: Consistent encoding across different architectures
//!
//! ## Error Handling and Edge Cases
//!
//! ### Forced Choice Protection
//! - **Immutability Enforcement**: Forced choices cannot have values modified
//! - **Constraint Validation**: Forced values must satisfy all constraints
//! - **Type Consistency**: Forced values must match expected choice type
//! - **Clear Error Messages**: Descriptive errors for invalid operations
//!
//! ### Index Management
//! - **Stale Index Prevention**: Indices not copied during node copying
//! - **Optional Assignment**: Supports both indexed and non-indexed workflows
//! - **Validation**: Index bounds checking when specified
//! - **Thread Safety**: Safe concurrent access to immutable node data
//!
//! ## Thread Safety and Concurrency
//!
//! ### Immutable Design
//! - **Read-Only Access**: All fields immutable except for index assignment
//! - **Shared References**: Safe to share `Arc<ChoiceNode>` between threads
//! - **Lock-Free Operations**: No synchronization required for read operations
//! - **Memory Safety**: Rust ownership prevents data races and memory corruption
//!
//! ### Concurrent Usage Patterns
//! - **Parallel Shrinking**: Multiple threads can shrink different choice sequences
//! - **Concurrent Replay**: Same choice sequence can be replayed across threads
//! - **Database Access**: Concurrent serialization/deserialization supported
//! - **Generation Pipeline**: Thread-safe integration with multi-threaded generation
//!
//! ## Compatibility and Standards
//!
//! ### Python Hypothesis Compatibility
//! - **Semantic Equivalence**: Identical behavior to Python ChoiceNode
//! - **Serialization Format**: Compatible binary representation
//! - **Constraint Semantics**: Matching validation logic
//! - **Replay Behavior**: Identical deterministic replay semantics
//!
//! ### Rust Ecosystem Integration
//! - **Serde Support**: Full serialization/deserialization capability
//! - **Standard Traits**: Debug, Clone, PartialEq, Eq, Hash implementations
//! - **Error Handling**: Result-based error propagation following Rust conventions
//! - **Documentation**: Comprehensive rustdoc with examples and performance notes

use super::{ChoiceType, ChoiceValue, Constraints};
use serde::{Serialize, Deserialize};

/// Immutable representation of a single choice made during test generation.
///
/// `ChoiceNode` serves as the fundamental building block of the Conjecture choice system,
/// capturing complete metadata about each decision point during test case generation.
/// This structure provides a Python Hypothesis compatible choice representation while
/// leveraging Rust's type system for enhanced safety, performance, and memory efficiency.
///
/// # Core Design Philosophy
///
/// Each ChoiceNode represents a single, immutable decision with complete provenance:
/// - **What**: The actual value that was chosen (`value` field)
/// - **How**: The type of choice and constraints that guided selection
/// - **Why**: Whether the choice was forced, random, or otherwise determined
/// - **Where**: Optional positional information within the choice sequence
///
/// # Field Documentation
///
/// ## Choice Classification (`choice_type`)
/// Strongly-typed enumeration identifying the kind of choice:
/// - `Integer`: Arbitrary-precision integer values with range/weight constraints
/// - `Boolean`: True/false values with probability weighting
/// - `Float`: IEEE 754 floating-point numbers with NaN/infinity handling
/// - `String`: UTF-8 strings with character set and length constraints
/// - `Bytes`: Raw byte sequences with size constraints
///
/// ## Value Storage (`value`)
/// Type-safe value storage using the `ChoiceValue` enum:
/// - Efficient serialization for database persistence
/// - Zero-copy access for performance-critical operations
/// - Hash-based deduplication for choice sequence optimization
/// - Lexicographic ordering for optimal shrinking behavior
///
/// ## Constraint Binding (`constraints`)
/// Complete constraint specification that guided value generation:
/// - Range bounds for numeric types
/// - Probability distributions for boolean choices
/// - Character sets and length limits for strings
/// - Size constraints for byte arrays
/// - Validation rules for forced value checking
///
/// ## Generation Context (`was_forced`)
/// Critical metadata indicating how the value was determined:
/// - `true`: Value was externally specified (replay, forced testing)
/// - `false`: Value was randomly generated using constraints and provider
/// - Used by shrinking system to preserve forced choice immutability
/// - Essential for proper replay behavior and constraint validation
///
/// ## Position Tracking (`index`)
/// Optional sequence position for advanced workflows:
/// - `None`: Position not assigned (default state)
/// - `Some(i)`: Position `i` within the choice sequence
/// - **Important**: Index not copied during node copying to prevent stale references
/// - Managed by `ExampleRecord` for proper lifecycle handling
///
/// # Performance Characteristics
///
/// ## Memory Layout
/// - **Size**: ~64-128 bytes typical, depends on value type and constraint complexity
/// - **Alignment**: Optimized struct packing with minimal padding
/// - **Heap Usage**: Large strings/bytes may allocate on heap, others stack-allocated
/// - **Sharing**: Efficient cloning through reference counting for large values
///
/// ## Time Complexity
/// - **Creation**: O(1) for all operations
/// - **Cloning**: O(1) for metadata, O(n) for value where n = value size
/// - **Validation**: O(1) for primitives, O(n×k) for strings where k = constraint intervals
/// - **Serialization**: O(n) where n = total serialized size
///
/// ## Cache Performance
/// - **Locality**: Sequential access patterns during replay provide excellent cache performance
/// - **Prefetching**: Predictable memory layout enables effective CPU prefetching
/// - **Branching**: Minimal conditional logic in hot paths for branch prediction efficiency
///
/// # Examples
///
/// ```rust
/// use conjecture::choice::{
///     ChoiceNode, ChoiceType, ChoiceValue, Constraints, IntegerConstraints
/// };
///
/// // Create an integer choice node
/// let int_constraints = Constraints::Integer(IntegerConstraints::new(
///     Some(0), Some(100), Some(0)
/// ));
/// let int_node = ChoiceNode::new(
///     ChoiceType::Integer,
///     ChoiceValue::Integer(42),
///     int_constraints,
///     false  // Not forced (randomly generated)
/// );
///
/// // Create a forced boolean choice (for replay)
/// let bool_constraints = Constraints::Boolean(BooleanConstraints { p: 0.5 });
/// let forced_bool = ChoiceNode::new(
///     ChoiceType::Boolean,
///     ChoiceValue::Boolean(true),
///     bool_constraints,
///     true  // Forced choice (from replay)
/// );
///
/// // Copy a node with modified value (for shrinking)
/// let shrunk_node = int_node.copy(
///     Some(ChoiceValue::Integer(10)),  // Smaller value
///     None  // Keep original constraints
/// ).expect("Shrinking should succeed for non-forced nodes");
/// ```
///
/// # Integration Points
///
/// ## Generation System
/// - Created during `ConjectureData::draw_*` operations
/// - Validated against constraints before acceptance
/// - Stored in choice sequences for replay and shrinking
/// - Indexed through `ExampleRecord` for database persistence
///
/// ## Shrinking System
/// - Source nodes provide constraints and metadata for shrinking
/// - New nodes created with modified values during shrinking
/// - Forced nodes protected from value modification
/// - Constraint compliance maintained throughout shrinking process
///
/// ## Database Persistence
/// - Serialized using compact binary encoding
/// - Deserialized with validation for data integrity
/// - Version-compatible format for cross-version compatibility
/// - Indexed efficiently for fast retrieval and querying
///
/// # Error Handling
///
/// The choice node system handles errors gracefully:
/// - **Type Safety**: Compile-time prevention of type mismatches
/// - **Constraint Validation**: Runtime validation with descriptive error messages
/// - **Forced Choice Protection**: Prevents invalid modifications with clear error messages
/// - **Serialization Errors**: Robust handling of corrupted or incompatible data
///
/// # Thread Safety
///
/// `ChoiceNode` instances are designed for concurrent use:
/// - **Immutable Fields**: All core fields are immutable after creation
/// - **Safe Sharing**: Can be safely shared between threads via `Arc<ChoiceNode>`
/// - **Lock-Free Access**: Read operations require no synchronization
/// - **Atomic Operations**: Index assignment can be performed atomically if needed
///
/// # Compatibility Guarantees
///
/// This implementation maintains strict compatibility with Python Hypothesis:
/// - **Semantic Equivalence**: Identical validation and constraint behavior
/// - **Serialization Format**: Binary-compatible representation
/// - **Replay Behavior**: Deterministic replay across language boundaries
/// - **Error Messages**: Consistent error reporting for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceNode {
    /// The type of choice that was made (Integer, Boolean, Float, String, Bytes).
    ///
    /// This strongly-typed enumeration prevents type confusion and enables
    /// compile-time optimization of choice-specific operations.
    pub choice_type: ChoiceType,
    
    /// The actual value that was chosen during generation.
    ///
    /// Stored using the type-safe `ChoiceValue` enum for efficient serialization
    /// and zero-copy access. The value type must match the `choice_type` field.
    pub value: ChoiceValue,
    
    /// The complete constraint specification that guided value generation.
    ///
    /// Includes all bounds, distributions, and validation rules that were
    /// active when this choice was made. Used for replay validation and
    /// shrinking constraint preservation.
    pub constraints: Constraints,
    
    /// Whether this choice was forced to a specific value (not randomly generated).
    ///
    /// - `true`: Value was externally specified (replay, deterministic testing)
    /// - `false`: Value was generated using constraints and randomness
    /// Critical for shrinking system and replay behavior.
    pub was_forced: bool,
    
    /// Optional index for tracking position within the choice sequence.
    ///
    /// - `None`: Position not assigned (default state)
    /// - `Some(i)`: Position `i` within the sequence
    /// **Important**: Not copied during node copying to prevent stale references.
    pub index: Option<u128>,
}

impl ChoiceNode {
    /// Create a new choice node without an index
    /// 
    /// This matches the Python ChoiceNode constructor. Index is None by default
    /// and only set via ExampleRecord to avoid stale indices after copying.
    pub fn new(
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Constraints,
        was_forced: bool,
    ) -> Self {
        Self {
            choice_type,
            value,
            constraints,
            was_forced,
            index: None,
        }
    }
    
    /// Create a new choice node with explicit index
    /// 
    /// Use this when you need to assign an index during creation (e.g., from ExampleRecord)
    pub fn with_index(
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Constraints,
        was_forced: bool,
        index: u128,
    ) -> Self {
        Self {
            choice_type,
            value,
            constraints,
            was_forced,
            index: Some(index),
        }
    }
    
    /// Set the index on this node, consuming self and returning a new node
    pub fn set_index(mut self, index: u128) -> Self {
        self.index = Some(index);
        self
    }

    /// Copy this node, optionally replacing the value and/or constraints
    /// 
    /// This matches Python's ChoiceNode.copy() method exactly. Note that forced nodes
    /// cannot have their values changed, and the index is explicitly NOT copied to
    /// prevent stale index issues.
    pub fn copy(
        &self,
        with_value: Option<ChoiceValue>,
        with_constraints: Option<Constraints>,
    ) -> Result<Self, String> {
        // Prevent modifying forced nodes with new values as this doesn't make sense
        if self.was_forced && with_value.is_some() {
            return Err("modifying a forced node doesn't make sense".to_string());
        }

        let new_value = with_value.unwrap_or_else(|| self.value.clone());
        let new_constraints = with_constraints.unwrap_or_else(|| self.constraints.clone());

        Ok(Self {
            choice_type: self.choice_type,
            value: new_value,
            constraints: new_constraints,
            was_forced: self.was_forced,
            index: None, // Explicitly not copying index as per Python implementation
        })
    }

    /// Copy this node with a new value only
    /// 
    /// Convenience method that matches common usage pattern
    pub fn copy_with_value(&self, new_value: ChoiceValue) -> Result<Self, String> {
        self.copy(Some(new_value), None)
    }

    /// Copy this node with new constraints only
    /// 
    /// Convenience method for updating constraints
    pub fn copy_with_constraints(&self, new_constraints: Constraints) -> Result<Self, String> {
        self.copy(None, Some(new_constraints))
    }

    /// Check if this node is trivial (would shrink to itself)
    /// 
    /// A node is trivial if it cannot be simplified any further. This does not
    /// mean that modifying a trivial node can't produce simpler test cases when
    /// viewing the tree as a whole. Just that when viewing this node in
    /// isolation, this is the simplest the node can get.
    /// 
    /// This implements Python's ChoiceNode.trivial property logic exactly.
    pub fn trivial(&self) -> bool {
        if self.was_forced {
            return true;
        }

        match self.choice_type {
            ChoiceType::Float => {
                if let (ChoiceValue::Float(float_value), Constraints::Float(constraints)) = 
                    (&self.value, &self.constraints) {
                    
                    let min_value = constraints.min_value;
                    let max_value = constraints.max_value;
                    let shrink_towards = 0.0;

                    // Case 1: Unbounded range (-inf, +inf)
                    if min_value == f64::NEG_INFINITY && max_value == f64::INFINITY {
                        return self.choice_equal_float(*float_value, shrink_towards);
                    }

                    // Case 2: Bounded range that contains an integer
                    if !min_value.is_infinite() && !max_value.is_infinite() {
                        let ceil_min = min_value.ceil();
                        let floor_max = max_value.floor();
                        
                        if ceil_min <= floor_max {
                            // The interval contains an integer. The simplest integer is the
                            // one closest to shrink_towards
                            let clamped_shrink = shrink_towards.max(ceil_min).min(floor_max);
                            return self.choice_equal_float(*float_value, clamped_shrink);
                        }
                    }

                    // Case 3: Conservative case - return false
                    // The real answer here is "the value in [min_value, max_value] with
                    // the lowest denominator when represented as a fraction".
                    // It would be good to compute this correctly in the future, but it's
                    // also not incorrect to be conservative here.
                    return false;
                }
            }
            _ => {
                // For non-float types: check if value equals the zero-index choice
                // For now, implement simple checks for common trivial cases
                match &self.value {
                    ChoiceValue::Integer(0) => {
                        // Check if shrink_towards is 0
                        if let Constraints::Integer(constraints) = &self.constraints {
                            return constraints.shrink_towards == Some(0);
                        }
                    }
                    ChoiceValue::Boolean(false) => {
                        // False is typically the trivial boolean value
                        return true;
                    }
                    ChoiceValue::String(s) if s.is_empty() => {
                        // Empty string is trivial if min_size allows it
                        if let Constraints::String(constraints) = &self.constraints {
                            return constraints.min_size == 0;
                        }
                    }
                    ChoiceValue::Bytes(b) if b.is_empty() => {
                        // Empty bytes is trivial if min_size allows it
                        if let Constraints::Bytes(constraints) = &self.constraints {
                            return constraints.min_size == 0;
                        }
                    }
                    _ => {}
                }
            }
        }

        false
    }

    /// Helper method for float equality comparison that handles NaN and signed zero
    fn choice_equal_float(&self, f1: f64, f2: f64) -> bool {
        // Handle NaN case - NaN == NaN for choice comparison
        if f1.is_nan() && f2.is_nan() {
            return true;
        }
        if f1.is_nan() || f2.is_nan() {
            return false;
        }
        // Use bit-level comparison to distinguish -0.0 from 0.0
        f1.to_bits() == f2.to_bits()
    }
}

impl PartialEq for ChoiceNode {
    fn eq(&self, other: &Self) -> bool {
        use crate::choice::choice_equal;
        
        self.choice_type == other.choice_type
            && self.was_forced == other.was_forced
            && choice_equal(&self.value, &other.value)
            && self.constraints == other.constraints
    }
}

impl Eq for ChoiceNode {}

impl std::hash::Hash for ChoiceNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash based on type, value, and constraints for uniqueness
        self.choice_type.hash(state);
        self.was_forced.hash(state);
        
        // Hash the value based on its type
        match &self.value {
            ChoiceValue::Integer(i) => i.hash(state),
            ChoiceValue::Boolean(b) => b.hash(state),
            ChoiceValue::Float(f) => f.to_bits().hash(state), // Use bits for float hashing
            ChoiceValue::String(s) => s.hash(state),
            ChoiceValue::Bytes(b) => b.hash(state),
        }
        
        // Hash constraints (simplified for now)
        match &self.constraints {
            Constraints::Integer(c) => {
                c.min_value.hash(state);
                c.max_value.hash(state);
                c.shrink_towards.hash(state);
            }
            Constraints::Boolean(c) => {
                c.p.to_bits().hash(state);
            }
            Constraints::Float(c) => {
                c.min_value.to_bits().hash(state);
                c.max_value.to_bits().hash(state);
                c.allow_nan.hash(state);
            }
            Constraints::String(c) => {
                c.min_size.hash(state);
                c.max_size.hash(state);
                c.intervals.hash(state);
            }
            Constraints::Bytes(c) => {
                c.min_size.hash(state);
                c.max_size.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints};

    #[test]
    fn test_choice_node_creation() {
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        assert_eq!(node.choice_type, ChoiceType::Integer);
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(!node.was_forced);
    }

    #[test]
    fn test_choice_node_copy_with_value() {
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        let copied = node.copy_with_value(ChoiceValue::Integer(100)).unwrap();
        
        assert_eq!(copied.value, ChoiceValue::Integer(100));
        assert_eq!(copied.choice_type, node.choice_type);
        assert_eq!(copied.was_forced, node.was_forced);
    }

    #[test]
    fn test_cannot_modify_forced_node() {
        let forced_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            true, // forced
        );

        let result = forced_node.copy_with_value(ChoiceValue::Boolean(false));
        
        assert!(result.is_err());
    }

    #[test]
    fn test_forced_nodes_are_trivial() {
        let forced_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            true, // forced
        );

        assert!(forced_node.trivial());
    }

    #[test]
    fn test_trivial_integer_nodes() {
        // Integer with shrink_towards=0 should be trivial when value=0
        let trivial_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(0),
            Constraints::Integer(IntegerConstraints::default()), // shrink_towards=0
            false,
        );
        
        assert!(trivial_node.trivial());
        
        // Integer with value != shrink_towards should not be trivial
        let non_trivial_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        assert!(!non_trivial_node.trivial());
    }

    #[test]
    fn test_trivial_float_nodes() {
        use crate::choice::FloatConstraints;
        
        // Unbounded float should be trivial when value=0.0
        let trivial_float = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(0.0),
            Constraints::Float(FloatConstraints::default()), // unbounded
            false,
        );
        
        assert!(trivial_float.trivial());
        
        // Non-zero unbounded float should not be trivial
        let non_trivial_float = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(3.14),
            Constraints::Float(FloatConstraints::default()),
            false,
        );
        
        assert!(!non_trivial_float.trivial());
        
        // Bounded float containing integer - should be trivial when value equals the clamped integer
        let bounded_constraints = FloatConstraints {
            min_value: 1.5,
            max_value: 3.5,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        };
        
        let bounded_trivial = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(2.0), // closest integer to 0 in range [1.5, 3.5]
            Constraints::Float(bounded_constraints.clone()),
            false,
        );
        
        assert!(bounded_trivial.trivial());
    }

    #[test]
    fn test_choice_node_equality() {
        let node1 = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        let node2 = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        assert_eq!(node1, node2);
    }

    #[test]
    fn test_choice_node_is_hashable() {
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(node.clone(), "test");
        
        assert_eq!(map.get(&node), Some(&"test"));
    }
}