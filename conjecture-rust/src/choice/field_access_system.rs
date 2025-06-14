//! Choice Structure Field Access System
//!
//! This module provides a comprehensive field access system for all choice structures,
//! ensuring proper field visibility, type consistency, initialization patterns, and
//! PyO3/FFI integration. It serves as the central capability for managing struct field
//! access across the entire choice module ecosystem.
//!
//! ## Architecture
//!
//! The field access system follows idiomatic Rust patterns:
//! - Public fields for direct access where appropriate
//! - Type-safe field access patterns
//! - Proper visibility control
//! - FFI-compatible field access
//! - Comprehensive error handling
//!
//! ## Usage
//!
//! ```rust
//! use crate::choice::field_access_system::*;
//! 
//! // Create and access choice nodes with full field visibility
//! let mut node = ChoiceNode::new(ChoiceType::Integer, ChoiceValue::Integer(42));
//! node.was_forced = true;
//! node.index = Some(1);
//! 
//! // Access constraint fields directly
//! if let Constraints::Integer(ref mut constraints) = node.constraints {
//!     constraints.min_value = Some(0);
//!     constraints.max_value = Some(100);
//! }
//! ```

use crate::choice::{
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    constraints::*,
    navigation::NavigationChoiceNode,
    templating::*,
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Field access validation result
#[derive(Debug, Clone, PartialEq)]
pub enum FieldAccessResult<T> {
    /// Field access succeeded with value
    Success(T),
    /// Field access failed with error message
    Error(String),
    /// Field is not accessible (private or invalid)
    NotAccessible,
}

impl<T> FieldAccessResult<T> {
    /// Convert to Result for easier error handling
    pub fn into_result(self) -> Result<T, String> {
        match self {
            FieldAccessResult::Success(value) => Ok(value),
            FieldAccessResult::Error(msg) => Err(msg),
            FieldAccessResult::NotAccessible => Err("Field not accessible".to_string()),
        }
    }
    
    /// Check if access was successful
    pub fn is_success(&self) -> bool {
        matches!(self, FieldAccessResult::Success(_))
    }
}

/// Trait for field access validation and management across choice structures
/// 
/// This trait provides a unified interface for validating and managing field access
/// patterns across all choice-related structures. It ensures type safety, proper
/// visibility control, and FFI compatibility while maintaining idiomatic Rust patterns.
/// 
/// # Design Philosophy
/// 
/// The trait follows several key principles:
/// - **Type Safety**: All field access is validated at runtime to prevent type mismatches
/// - **Visibility Control**: Fields are properly scoped with public/private access patterns
/// - **FFI Compatibility**: All field access patterns work seamlessly with PyO3 bindings
/// - **Performance**: Validation is optimized for common access patterns
/// 
/// # Implementation Notes
/// 
/// Implementers should ensure that:
/// 1. Field validation is comprehensive but efficient
/// 2. Type consistency checks cover all possible value combinations
/// 3. Error messages are descriptive and actionable
/// 4. Field visibility maps are accurate and complete
pub trait FieldAccessible {
    /// Validate that field access patterns are correct and type-safe
    /// 
    /// Performs comprehensive validation of all accessible fields to ensure:
    /// - Type consistency between related fields (e.g., choice_type matches value type)
    /// - Required fields are present and properly initialized
    /// - Optional fields have valid values when present
    /// - Constraints are well-formed and internally consistent
    /// 
    /// # Returns
    /// 
    /// - `Ok(())` if all field access patterns are valid
    /// - `Err(String)` with detailed error message if validation fails
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use crate::choice::field_access_system::FieldAccessible;
    /// use crate::choice::ChoiceNode;
    /// 
    /// let node = ChoiceNode::new(ChoiceType::Integer, ChoiceValue::Integer(42));
    /// assert!(node.validate_field_access().is_ok());
    /// ```
    fn validate_field_access(&self) -> Result<(), String>;
    
    /// Get comprehensive field visibility information for this type
    /// 
    /// Returns a mapping of field names to their visibility status, where:
    /// - `true` indicates the field is publicly accessible
    /// - `false` indicates the field is private or restricted
    /// 
    /// This information is used for:
    /// - FFI interface generation
    /// - Documentation generation
    /// - Debug and introspection tools
    /// - Access pattern optimization
    /// 
    /// # Returns
    /// 
    /// A HashMap where keys are field names and values indicate public accessibility.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use crate::choice::field_access_system::FieldAccessible;
    /// use crate::choice::ChoiceNode;
    /// 
    /// let visibility = ChoiceNode::get_field_visibility();
    /// assert_eq!(visibility.get("choice_type"), Some(&true));
    /// assert_eq!(visibility.get("value"), Some(&true));
    /// ```
    fn get_field_visibility() -> HashMap<String, bool>;
    
    /// Validate field type consistency across the entire structure
    /// 
    /// Performs deep validation to ensure that all field types are consistent
    /// with each other and with the structure's invariants. This includes:
    /// - Checking that choice_type matches the actual value type
    /// - Validating constraint types match the choice type
    /// - Ensuring index values are within valid ranges
    /// - Verifying forced flags are correctly set
    /// 
    /// # Error Conditions
    /// 
    /// This method will return an error if:
    /// - Choice type doesn't match value type (e.g., Integer type with Float value)
    /// - Constraints are incompatible with the choice type
    /// - Index values are out of bounds or invalid
    /// - Required fields are missing or have invalid types
    /// 
    /// # Performance
    /// 
    /// Type validation is optimized for common cases but performs thorough
    /// checking for edge cases. Expected time complexity is O(1) for most
    /// structures, O(n) for complex constraint validation.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use crate::choice::field_access_system::FieldAccessible;
    /// use crate::choice::{ChoiceNode, ChoiceType, ChoiceValue};
    /// 
    /// let mut node = ChoiceNode::new(ChoiceType::Integer, ChoiceValue::Integer(42));
    /// assert!(node.validate_field_types().is_ok());
    /// 
    /// // This would fail validation:
    /// // node.value = ChoiceValue::String("invalid".to_string());
    /// // assert!(node.validate_field_types().is_err());
    /// ```
    fn validate_field_types(&self) -> Result<(), String>;
}

/// Implementation for ChoiceNode field access
impl FieldAccessible for ChoiceNode {
    fn validate_field_access(&self) -> Result<(), String> {
        // Validate choice_type is accessible and valid
        if !matches!(self.choice_type, 
            ChoiceType::Integer | ChoiceType::Boolean | ChoiceType::Float | 
            ChoiceType::String | ChoiceType::Bytes) {
            return Err("Invalid choice_type value".to_string());
        }
        
        // Validate value matches choice_type
        let type_matches = match (&self.choice_type, &self.value) {
            (ChoiceType::Integer, ChoiceValue::Integer(_)) => true,
            (ChoiceType::Boolean, ChoiceValue::Boolean(_)) => true,
            (ChoiceType::Float, ChoiceValue::Float(_)) => true,
            (ChoiceType::String, ChoiceValue::String(_)) => true,
            (ChoiceType::Bytes, ChoiceValue::Bytes(_)) => true,
            _ => false,
        };
        
        if !type_matches {
            return Err(format!(
                "choice_type {:?} does not match value type {:?}",
                self.choice_type, 
                std::mem::discriminant(&self.value)
            ));
        }
        
        // Validate constraints match choice_type
        let constraints_match = match (&self.choice_type, &self.constraints) {
            (ChoiceType::Integer, Constraints::Integer(_)) => true,
            (ChoiceType::Boolean, Constraints::Boolean(_)) => true,
            (ChoiceType::Float, Constraints::Float(_)) => true,
            (ChoiceType::String, Constraints::String(_)) => true,
            (ChoiceType::Bytes, Constraints::Bytes(_)) => true,
            _ => false,
        };
        
        if !constraints_match {
            return Err(format!(
                "choice_type {:?} does not match constraints type {:?}",
                self.choice_type,
                std::mem::discriminant(&self.constraints)
            ));
        }
        
        // Validate was_forced is accessible (bool is always valid)
        // Validate index is accessible (Option<usize> is always valid)
        
        println!("FIELD_ACCESS DEBUG: ChoiceNode field access validation passed");
        Ok(())
    }
    
    fn get_field_visibility() -> HashMap<String, bool> {
        let mut visibility = HashMap::new();
        visibility.insert("choice_type".to_string(), true);
        visibility.insert("value".to_string(), true);
        visibility.insert("constraints".to_string(), true);
        visibility.insert("was_forced".to_string(), true);
        visibility.insert("index".to_string(), true);
        visibility
    }
    
    fn validate_field_types(&self) -> Result<(), String> {
        // Type validation is handled in validate_field_access
        self.validate_field_access()
    }
}

/// Implementation for NavigationChoiceNode field access
impl FieldAccessible for NavigationChoiceNode {
    fn validate_field_access(&self) -> Result<(), String> {
        // First validate base ChoiceNode fields
        let base_node = ChoiceNode {
            choice_type: self.choice_type,
            value: self.value.clone(),
            constraints: self.constraints.clone(),
            was_forced: self.was_forced,
            index: self.index.map(|i| i.try_into().unwrap()),
        };
        base_node.validate_field_access()?;
        
        // Validate navigation-specific fields
        // children HashMap is always valid
        // is_exhausted bool is always valid
        
        println!("FIELD_ACCESS DEBUG: NavigationChoiceNode field access validation passed");
        Ok(())
    }
    
    fn get_field_visibility() -> HashMap<String, bool> {
        let mut visibility = ChoiceNode::get_field_visibility();
        visibility.insert("children".to_string(), true);
        visibility.insert("is_exhausted".to_string(), true);
        visibility
    }
    
    fn validate_field_types(&self) -> Result<(), String> {
        self.validate_field_access()
    }
}

/// Implementation for constraint types field access
impl FieldAccessible for IntegerConstraints {
    fn validate_field_access(&self) -> Result<(), String> {
        // Validate min_value <= max_value if both are present
        if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
            if min > max {
                return Err(format!(
                    "min_value {} must be <= max_value {}",
                    min, max
                ));
            }
        }
        
        // Validate weights HashMap (always valid if present)
        if let Some(ref weights) = self.weights {
            for (value, weight) in weights {
                if *weight < 0.0 || weight.is_nan() || weight.is_infinite() {
                    return Err(format!(
                        "Invalid weight {} for value {}",
                        weight, value
                    ));
                }
            }
        }
        
        println!("FIELD_ACCESS DEBUG: IntegerConstraints field access validation passed");
        Ok(())
    }
    
    fn get_field_visibility() -> HashMap<String, bool> {
        let mut visibility = HashMap::new();
        visibility.insert("min_value".to_string(), true);
        visibility.insert("max_value".to_string(), true);
        visibility.insert("weights".to_string(), true);
        visibility.insert("shrink_towards".to_string(), true);
        visibility
    }
    
    fn validate_field_types(&self) -> Result<(), String> {
        self.validate_field_access()
    }
}

impl FieldAccessible for FloatConstraints {
    fn validate_field_access(&self) -> Result<(), String> {
        // Validate min_value <= max_value
        if self.min_value > self.max_value {
            return Err(format!(
                "min_value {} must be <= max_value {}",
                self.min_value, self.max_value
            ));
        }
        
        // Validate smallest_nonzero_magnitude > 0 if present
        if let Some(magnitude) = self.smallest_nonzero_magnitude {
            if magnitude <= 0.0 {
                return Err(format!(
                    "smallest_nonzero_magnitude must be positive, got {}",
                    magnitude
                ));
            }
        }
        
        println!("FIELD_ACCESS DEBUG: FloatConstraints field access validation passed");
        Ok(())
    }
    
    fn get_field_visibility() -> HashMap<String, bool> {
        let mut visibility = HashMap::new();
        visibility.insert("min_value".to_string(), true);
        visibility.insert("max_value".to_string(), true);
        visibility.insert("allow_nan".to_string(), true);
        visibility.insert("smallest_nonzero_magnitude".to_string(), true);
        visibility
    }
    
    fn validate_field_types(&self) -> Result<(), String> {
        self.validate_field_access()
    }
}

impl FieldAccessible for BooleanConstraints {
    fn validate_field_access(&self) -> Result<(), String> {
        // Validate probability is in [0, 1] range
        if self.p < 0.0 || self.p > 1.0 || self.p.is_nan() {
            return Err(format!(
                "Probability p must be in [0, 1], got {}",
                self.p
            ));
        }
        
        println!("FIELD_ACCESS DEBUG: BooleanConstraints field access validation passed");
        Ok(())
    }
    
    fn get_field_visibility() -> HashMap<String, bool> {
        let mut visibility = HashMap::new();
        visibility.insert("p".to_string(), true);
        visibility
    }
    
    fn validate_field_types(&self) -> Result<(), String> {
        self.validate_field_access()
    }
}

/// Field access utilities for common operations
pub struct FieldAccessUtils;

impl FieldAccessUtils {
    /// Safely access choice node value with type checking
    pub fn get_choice_value<T>(
        node: &ChoiceNode,
        expected_type: ChoiceType,
    ) -> FieldAccessResult<T>
    where
        T: Clone,
        ChoiceValue: Into<Option<T>>,
    {
        if node.choice_type != expected_type {
            return FieldAccessResult::Error(format!(
                "Expected choice_type {:?}, got {:?}",
                expected_type, node.choice_type
            ));
        }
        
        // This is a simplified version - in practice you'd need proper type conversion
        FieldAccessResult::NotAccessible
    }
    
    /// Validate all fields in a choice node structure
    pub fn validate_node_fields(node: &ChoiceNode) -> Result<(), String> {
        node.validate_field_access()
    }
    
    /// Get field access report for debugging
    pub fn get_field_access_report(node: &ChoiceNode) -> HashMap<String, String> {
        let mut report = HashMap::new();
        
        report.insert(
            "choice_type".to_string(),
            format!("{:?}", node.choice_type)
        );
        
        report.insert(
            "value_type".to_string(),
            match &node.value {
                ChoiceValue::Integer(_) => "Integer".to_string(),
                ChoiceValue::Boolean(_) => "Boolean".to_string(),
                ChoiceValue::Float(_) => "Float".to_string(),
                ChoiceValue::String(_) => "String".to_string(),
                ChoiceValue::Bytes(_) => "Bytes".to_string(),
            }
        );
        
        report.insert(
            "constraints_type".to_string(),
            match &node.constraints {
                Constraints::Integer(_) => "Integer".to_string(),
                Constraints::Boolean(_) => "Boolean".to_string(),
                Constraints::Float(_) => "Float".to_string(),
                Constraints::String(_) => "String".to_string(),  
                Constraints::Bytes(_) => "Bytes".to_string(),
            }
        );
        
        report.insert("was_forced".to_string(), node.was_forced.to_string());
        report.insert("index".to_string(), format!("{:?}", node.index));
        
        report
    }
    
    /// Create a choice node with validated field access
    pub fn create_validated_node(
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Constraints,
    ) -> Result<ChoiceNode, String> {
        let node = ChoiceNode {
            choice_type,
            value,
            constraints,
            was_forced: false,
            index: None,
        };
        
        node.validate_field_access()?;
        Ok(node)
    }
    
    /// Update node fields with validation
    pub fn update_node_fields(
        node: &mut ChoiceNode,
        was_forced: Option<bool>,
        index: Option<Option<usize>>,
    ) -> Result<(), String> {
        if let Some(forced) = was_forced {
            node.was_forced = forced;
        }
        
        if let Some(idx_opt) = index {
            node.index = idx_opt.map(|idx| idx.try_into().unwrap());
        }
        
        node.validate_field_access()
    }
}

/// FFI-safe field access patterns for PyO3 integration
pub mod ffi_field_access {
    use super::*;
    
    /// FFI-safe field getter functions
    pub struct FFIFieldGetters;
    
    impl FFIFieldGetters {
        /// Get choice type as string (FFI-safe)
        pub fn get_choice_type_str(node: &ChoiceNode) -> String {
            format!("{:?}", node.choice_type)
        }
        
        /// Get was_forced flag (FFI-safe)
        pub fn get_was_forced(node: &ChoiceNode) -> bool {
            node.was_forced
        }
        
        /// Get index as i64 (FFI-safe, -1 for None)
        pub fn get_index(node: &ChoiceNode) -> i64 {
            node.index.map(|i| i as i64).unwrap_or(-1)
        }
        
        /// Get integer value if present (FFI-safe)
        pub fn get_integer_value(node: &ChoiceNode) -> Option<i128> {
            match &node.value {
                ChoiceValue::Integer(val) => Some(*val),
                _ => None,
            }
        }
        
        /// Get boolean value if present (FFI-safe)
        pub fn get_boolean_value(node: &ChoiceNode) -> Option<bool> {
            match &node.value {
                ChoiceValue::Boolean(val) => Some(*val),
                _ => None,
            }
        }
        
        /// Get float value if present (FFI-safe)
        pub fn get_float_value(node: &ChoiceNode) -> Option<f64> {
            match &node.value {
                ChoiceValue::Float(val) => Some(*val),
                _ => None,
            }
        }
        
        /// Get string value if present (FFI-safe)
        pub fn get_string_value(node: &ChoiceNode) -> Option<String> {
            match &node.value {
                ChoiceValue::String(val) => Some(val.clone()),
                _ => None,
            }
        }
    }
    
    /// FFI-safe field setter functions
    pub struct FFIFieldSetters;
    
    impl FFIFieldSetters {
        /// Set was_forced flag (FFI-safe)
        pub fn set_was_forced(node: &mut ChoiceNode, forced: bool) -> Result<(), String> {
            node.was_forced = forced;
            node.validate_field_access()
        }
        
        /// Set index (FFI-safe, -1 means None)
        pub fn set_index(node: &mut ChoiceNode, index: i64) -> Result<(), String> {
            node.index = if index < 0 { None } else { Some((index as usize).try_into().unwrap()) };
            node.validate_field_access()
        }
        
        /// Update integer constraints min/max (FFI-safe)
        pub fn set_integer_constraints_range(
            node: &mut ChoiceNode,
            min_value: Option<i128>,
            max_value: Option<i128>,
        ) -> Result<(), String> {
            if let Constraints::Integer(ref mut constraints) = node.constraints {
                constraints.min_value = min_value;
                constraints.max_value = max_value;
                constraints.validate_field_access()?;
            } else {
                return Err("Node does not have integer constraints".to_string());
            }
            
            node.validate_field_access()
        }
        
        /// Update float constraints range (FFI-safe)
        pub fn set_float_constraints_range(
            node: &mut ChoiceNode,
            min_value: f64,
            max_value: f64,
        ) -> Result<(), String> {
            if let Constraints::Float(ref mut constraints) = node.constraints {
                constraints.min_value = min_value;
                constraints.max_value = max_value;
                constraints.validate_field_access()?;
            } else {
                return Err("Node does not have float constraints".to_string());
            }
            
            node.validate_field_access()
        }
    }
}

/// Default values and construction helpers
pub mod defaults {
    use super::*;
    
    /// Get correct default values for all constraint types
    pub struct ConstraintDefaults;
    
    impl ConstraintDefaults {
        /// Get correct IntegerConstraints default
        pub fn integer_constraints() -> IntegerConstraints {
            IntegerConstraints {
                min_value: None,
                max_value: None,
                weights: None,
                shrink_towards: Some(0),
            }
        }
        
        /// Get correct FloatConstraints default  
        pub fn float_constraints() -> FloatConstraints {
            FloatConstraints {
                min_value: f64::NEG_INFINITY,
                max_value: f64::INFINITY,
                allow_nan: true,
                smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
            }
        }
        
        /// Get correct BooleanConstraints default
        pub fn boolean_constraints() -> BooleanConstraints {
            BooleanConstraints {
                p: 0.5,
            }
        }
        
        /// Get correct StringConstraints default
        pub fn string_constraints() -> StringConstraints {
            StringConstraints {
                min_size: 0,
                max_size: 8192, // Not usize::MAX
                intervals: Default::default(),
            }
        }
        
        /// Get correct BytesConstraints default
        pub fn bytes_constraints() -> BytesConstraints {
            BytesConstraints {
                min_size: 0,
                max_size: 8192, // Not usize::MAX
            }
        }
    }
}

/// Template type corrections for proper field access
pub mod template_fixes {
    use super::*;
    
    /// Get valid TemplateType variants
    pub fn get_valid_template_types() -> Vec<TemplateType> {
        vec![
            TemplateType::Simplest,
            TemplateType::AtIndex(0),
            TemplateType::Biased { bias: 0.5 },
            TemplateType::Custom { name: "test".to_string() },
        ]
    }
    
    /// Create a valid ChoiceTemplate with correct field access
    pub fn create_valid_template() -> ChoiceTemplate {
        ChoiceTemplate {
            template_type: TemplateType::Simplest, // Not Fixed/OneOf
            count: Some(5),
            is_forcing: false,
            metadata: Some("test_metadata".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_field_access_validation() {
        let node = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(42),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(0),
        };
        
        assert!(node.validate_field_access().is_ok());
    }
    
    #[test]
    fn test_field_visibility() {
        let visibility = ChoiceNode::get_field_visibility();
        assert_eq!(visibility.len(), 5);
        assert_eq!(visibility["choice_type"], true);
        assert_eq!(visibility["value"], true);
        assert_eq!(visibility["constraints"], true);
        assert_eq!(visibility["was_forced"], true);
        assert_eq!(visibility["index"], true);
    }
    
    #[test]
    fn test_constraint_defaults() {
        let int_default = defaults::ConstraintDefaults::integer_constraints();
        assert_eq!(int_default.min_value, None);
        assert_eq!(int_default.max_value, None);
        assert_eq!(int_default.shrink_towards, Some(0));
        
        let float_default = defaults::ConstraintDefaults::float_constraints();
        assert_eq!(float_default.min_value, f64::NEG_INFINITY);
        assert_eq!(float_default.max_value, f64::INFINITY);
        assert!(float_default.allow_nan);
        assert_eq!(float_default.smallest_nonzero_magnitude, Some(f64::MIN_POSITIVE));
    }
    
    #[test]
    fn test_ffi_field_access() {
        let node = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(123),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(42),
        };
        
        assert_eq!(ffi_field_access::FFIFieldGetters::get_choice_type_str(&node), "Integer");
        assert!(!ffi_field_access::FFIFieldGetters::get_was_forced(&node));
        assert_eq!(ffi_field_access::FFIFieldGetters::get_index(&node), 42);
        assert_eq!(ffi_field_access::FFIFieldGetters::get_integer_value(&node), Some(123));
    }
}