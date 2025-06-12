//! Choice templating and forcing system for guided test case construction
//!
//! This module implements template-driven choice generation and forced value insertion
//! that mirrors Python Hypothesis's templating capabilities. It provides sophisticated
//! mechanisms for constructing specific test case patterns while maintaining full
//! compatibility with the choice constraint system.

use crate::choice::{ChoiceType, ChoiceValue, Constraints, ChoiceNode};
use std::collections::VecDeque;

/// Template type for generating choices with specific patterns
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateType {
    /// Generate the simplest possible choice (index 0)
    Simplest,
    /// Generate a choice at specific index
    AtIndex(usize),
    /// Generate choice with specific bias/probability
    Biased { bias: f64 },
    /// Custom template with user-defined generation logic
    Custom { name: String },
}

impl Eq for TemplateType {}

impl std::hash::Hash for TemplateType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            TemplateType::Simplest => {
                0u8.hash(state);
            }
            TemplateType::AtIndex(idx) => {
                1u8.hash(state);
                idx.hash(state);
            }
            TemplateType::Biased { bias } => {
                2u8.hash(state);
                bias.to_bits().hash(state);
            }
            TemplateType::Custom { name } => {
                3u8.hash(state);
                name.hash(state);
            }
        }
    }
}

impl Default for TemplateType {
    fn default() -> Self {
        TemplateType::Simplest
    }
}

impl std::fmt::Display for TemplateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateType::Simplest => write!(f, "simplest"),
            TemplateType::AtIndex(idx) => write!(f, "at_index({})", idx),
            TemplateType::Biased { bias } => write!(f, "biased({:.3})", bias),
            TemplateType::Custom { name } => write!(f, "custom({})", name),
        }
    }
}

/// Template for generating choices with specific patterns
/// 
/// Mirrors Python's ChoiceTemplate class but with Rust's type safety and ownership model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceTemplate {
    /// The type of template generation to use
    pub template_type: TemplateType,
    
    /// Optional count limit for how many times this template can be used
    pub count: Option<usize>,
    
    /// Whether this template should force choices (bypass randomness)
    pub is_forcing: bool,
    
    /// Optional metadata for template debugging and analysis
    pub metadata: Option<String>,
}

impl ChoiceTemplate {
    /// Create a new template that generates the simplest choice
    pub fn simplest() -> Self {
        Self {
            template_type: TemplateType::Simplest,
            count: None, // Single-use by default
            is_forcing: true,
            metadata: None,
        }
    }
    
    /// Create a template that generates choices with unlimited usage
    pub fn unlimited(template_type: TemplateType) -> Self {
        Self {
            template_type,
            count: Some(usize::MAX), // Use max value to represent unlimited
            is_forcing: true,
            metadata: None,
        }
    }
    
    /// Create a template with limited usage count
    pub fn with_count(template_type: TemplateType, count: usize) -> Self {
        Self {
            template_type,
            count: Some(count),
            is_forcing: true,
            metadata: None,
        }
    }
    
    /// Create a template at specific index
    pub fn at_index(index: usize) -> Self {
        Self {
            template_type: TemplateType::AtIndex(index),
            count: None, // Single-use by default
            is_forcing: true,
            metadata: Some(format!("Index template for position {}", index)),
        }
    }
    
    /// Create a biased template with specified probability bias
    pub fn biased(bias: f64) -> Self {
        Self {
            template_type: TemplateType::Biased { bias },
            count: None, // Single-use by default
            is_forcing: false, // Biased templates still use randomness
            metadata: Some(format!("Biased template with bias {:.3}", bias)),
        }
    }
    
    /// Create a custom template with user-defined logic
    pub fn custom(name: String) -> Self {
        Self {
            template_type: TemplateType::Custom { name: name.clone() },
            count: None, // Single-use by default
            is_forcing: true,
            metadata: Some(format!("Custom template: {}", name)),
        }
    }
    
    /// Set metadata for debugging and analysis
    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    /// Check if this template has remaining usage count
    pub fn has_remaining_count(&self) -> bool {
        match self.count {
            Some(0) => false,
            _ => true,
        }
    }
    
    /// Consume one usage of this template, returning updated template
    pub fn consume_usage(mut self) -> Result<Self, TemplateError> {
        match self.count {
            Some(0) => Err(TemplateError::ExhaustedTemplate),
            Some(count) => {
                if count == usize::MAX {
                    // Unlimited template, don't decrement
                    Ok(self)
                } else {
                    self.count = Some(count - 1);
                    Ok(self)
                }
            }
            None => {
                // Single-use template, mark as consumed
                self.count = Some(0);
                Ok(self)
            }
        }
    }
    
    /// Get remaining usage count (None for unlimited templates)
    pub fn remaining_count(&self) -> Option<usize> {
        match self.count {
            Some(usize::MAX) => None, // Unlimited templates return None
            other => other,
        }
    }
}

impl std::hash::Hash for ChoiceTemplate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.template_type.hash(state);
        self.count.hash(state);
        self.is_forcing.hash(state);
        self.metadata.hash(state);
    }
}

/// Template entries can be either templates or direct values
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateEntry {
    /// A template that generates choices with specific patterns
    Template(ChoiceTemplate),
    /// A direct value to force as the choice
    DirectValue(ChoiceValue),
    /// A partial node specification (value + constraints)
    PartialNode {
        value: ChoiceValue,
        constraints: Option<Constraints>,
        metadata: Option<String>,
    },
}

impl TemplateEntry {
    /// Create a template entry from a direct value
    pub fn direct(value: ChoiceValue) -> Self {
        Self::DirectValue(value)
    }
    
    /// Create a template entry from a template
    pub fn template(template: ChoiceTemplate) -> Self {
        Self::Template(template)
    }
    
    /// Create a partial node entry with value and optional constraints
    pub fn partial_node(
        value: ChoiceValue,
        constraints: Option<Constraints>,
        metadata: Option<String>,
    ) -> Self {
        Self::PartialNode {
            value,
            constraints,
            metadata,
        }
    }
    
    /// Check if this entry forces a choice (bypasses randomness)
    pub fn is_forcing(&self) -> bool {
        match self {
            Self::Template(template) => template.is_forcing,
            Self::DirectValue(_) => true,
            Self::PartialNode { .. } => true,
        }
    }
    
    /// Get debug description of this entry
    pub fn debug_description(&self) -> String {
        match self {
            Self::Template(template) => format!("Template({})", template.template_type),
            Self::DirectValue(value) => format!("Direct({:?})", value),
            Self::PartialNode { value, constraints, metadata } => {
                let constraints_desc = constraints.as_ref()
                    .map(|_| format!(" with constraints"))
                    .unwrap_or_default();
                let metadata_desc = metadata.as_ref()
                    .map(|m| format!(" ({})", m))
                    .unwrap_or_default();
                format!("Partial({:?}{}{})", value, constraints_desc, metadata_desc)
            }
        }
    }
}

/// Template processing engine that handles template-driven choice generation
#[derive(Debug, Clone)]
pub struct TemplateEngine {
    /// Queue of template entries to process
    template_queue: VecDeque<TemplateEntry>,
    
    /// Track how many templates have been processed
    processed_count: usize,
    
    /// Track misalignment issues between templates and actual constraints
    misalignment_index: Option<usize>,
    
    /// Enable debug logging for template processing
    pub debug_mode: bool,
    
    /// Metadata for template engine analysis
    pub metadata: Option<String>,
}

impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> Self {
        Self {
            template_queue: VecDeque::new(),
            processed_count: 0,
            misalignment_index: None,
            debug_mode: false,
            metadata: None,
        }
    }
    
    /// Create template engine from a sequence of template entries
    pub fn from_entries(entries: Vec<TemplateEntry>) -> Self {
        let mut engine = Self::new();
        for entry in entries {
            engine.add_entry(entry);
        }
        // Created template engine with entries
        engine
    }
    
    /// Create template engine from templates only
    pub fn from_templates(templates: Vec<ChoiceTemplate>) -> Self {
        let entries: Vec<TemplateEntry> = templates
            .into_iter()
            .map(TemplateEntry::Template)
            .collect();
        Self::from_entries(entries)
    }
    
    /// Create template engine from direct values only
    pub fn from_values(values: Vec<ChoiceValue>) -> Self {
        let entries: Vec<TemplateEntry> = values
            .into_iter()
            .map(TemplateEntry::DirectValue)
            .collect();
        Self::from_entries(entries)
    }
    
    /// Enable debug mode for detailed logging
    pub fn with_debug(mut self) -> Self {
        self.debug_mode = true;
        self
    }
    
    /// Set metadata for engine analysis
    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    /// Add a template entry to the processing queue
    pub fn add_entry(&mut self, entry: TemplateEntry) {
        if self.debug_mode {
            // Adding template entry
        }
        self.template_queue.push_back(entry);
    }
    
    /// Add multiple template entries
    pub fn add_entries(&mut self, entries: Vec<TemplateEntry>) {
        for entry in entries {
            self.add_entry(entry);
        }
    }
    
    /// Check if there are remaining templates to process
    pub fn has_templates(&self) -> bool {
        !self.template_queue.is_empty()
    }
    
    /// Get the number of remaining templates
    pub fn remaining_count(&self) -> usize {
        self.template_queue.len()
    }
    
    /// Get the number of processed templates
    pub fn processed_count(&self) -> usize {
        self.processed_count
    }
    
    /// Check if there was a misalignment during template processing
    pub fn has_misalignment(&self) -> bool {
        self.misalignment_index.is_some()
    }
    
    /// Get the index where misalignment first occurred
    pub fn misalignment_index(&self) -> Option<usize> {
        self.misalignment_index
    }
    
    /// Process the next template in the queue for the given choice type and constraints
    /// 
    /// Returns the generated choice node or None if no templates remain.
    /// Handles misalignment by falling back to simplest valid choice.
    pub fn process_next_template(
        &mut self,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<Option<ChoiceNode>, TemplateError> {
        if let Some(entry) = self.template_queue.pop_front() {
            self.processed_count += 1;
            
            if self.debug_mode {
                // Processing template entry
            }
            
            match entry {
                TemplateEntry::DirectValue(value) => {
                    self.process_direct_value(value, choice_type, constraints)
                }
                TemplateEntry::Template(template) => {
                    self.process_template(template, choice_type, constraints)
                }
                TemplateEntry::PartialNode { value, constraints: node_constraints, metadata } => {
                    self.process_partial_node(
                        value,
                        node_constraints,
                        metadata,
                        choice_type,
                        constraints,
                    )
                }
            }
        } else {
            Ok(None)
        }
    }
    
    /// Process a direct value entry
    fn process_direct_value(
        &mut self,
        value: ChoiceValue,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<Option<ChoiceNode>, TemplateError> {
        // Validate value type matches expected choice type
        if !self.value_matches_type(&value, choice_type) {
            // Template value type mismatch
            self.record_misalignment();
            return self.fallback_to_simplest(choice_type, constraints);
        }
        
        // Validate value satisfies constraints
        if !self.value_satisfies_constraints(&value, constraints) {
            // Template value violates constraints
            self.record_misalignment();
            return self.fallback_to_simplest(choice_type, constraints);
        }
        
        // Successfully processed direct value
        
        Ok(Some(ChoiceNode::new(
            choice_type,
            value,
            constraints.clone(),
            true, // Direct values are always forced
        )))
    }
    
    /// Process a template entry
    fn process_template(
        &mut self,
        mut template: ChoiceTemplate,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<Option<ChoiceNode>, TemplateError> {
        // Check if template has remaining uses
        if !template.has_remaining_count() {
            // Template has no remaining uses
            return Err(TemplateError::ExhaustedTemplate);
        }
        
        // Consume one use of the template
        template = template.consume_usage()?;
        
        // If template still has remaining uses, re-add it to the back of the queue
        if template.has_remaining_count() {
            self.template_queue.push_back(TemplateEntry::Template(template.clone()));
        }
        
        // Generate choice based on template type
        let choice_result = match &template.template_type {
            TemplateType::Simplest => {
                // Generating simplest choice
                self.generate_simplest_choice(choice_type, constraints)
            }
            TemplateType::AtIndex(index) => {
                // Generating choice at index
                self.generate_choice_at_index(*index, choice_type, constraints)
            }
            TemplateType::Biased { bias } => {
                // Generating biased choice
                self.generate_biased_choice(*bias, choice_type, constraints)
            }
            TemplateType::Custom { name } => {
                // Processing custom template
                self.generate_custom_choice(name, choice_type, constraints)
            }
        };
        
        match choice_result {
            Ok(node) => {
                // For template-generated choices, keep the original was_forced value
                // Only override for direct value forcing, not template-based generation
                
                // Successfully processed template
                
                Ok(Some(node))
            }
            Err(_e) => {
                // Template processing failed
                self.record_misalignment();
                self.fallback_to_simplest(choice_type, constraints)
            }
        }
    }
    
    /// Process a partial node entry
    fn process_partial_node(
        &mut self,
        value: ChoiceValue,
        node_constraints: Option<Constraints>,
        metadata: Option<String>,
        choice_type: ChoiceType,
        expected_constraints: &Constraints,
    ) -> Result<Option<ChoiceNode>, TemplateError> {
        // Use provided constraints or fall back to expected constraints
        let final_constraints = node_constraints.unwrap_or_else(|| expected_constraints.clone());
        
        // Validate value type and constraints
        if !self.value_matches_type(&value, choice_type) {
            // Partial node value type mismatch
            self.record_misalignment();
            return self.fallback_to_simplest(choice_type, expected_constraints);
        }
        
        if !self.value_satisfies_constraints(&value, &final_constraints) {
            // Partial node value violates constraints
            self.record_misalignment();
            return self.fallback_to_simplest(choice_type, expected_constraints);
        }
        
        if let Some(_meta) = metadata {
            // Processing partial node with metadata
        }
        
        Ok(Some(ChoiceNode::new(
            choice_type,
            value,
            final_constraints,
            true, // Partial nodes are always forced
        )))
    }
    
    /// Generate the simplest valid choice for the given type and constraints
    pub fn generate_simplest_choice(
        &self,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceNode, TemplateError> {
        let value = match choice_type {
            ChoiceType::Integer => {
                if let Constraints::Integer(int_constraints) = constraints {
                    let shrink_towards = int_constraints.shrink_towards.unwrap_or(0);
                    let min_val = int_constraints.min_value.unwrap_or(i128::MIN);
                    let max_val = int_constraints.max_value.unwrap_or(i128::MAX);
                    let clamped = shrink_towards.max(min_val).min(max_val);
                    ChoiceValue::Integer(clamped)
                } else {
                    return Err(TemplateError::ConstraintMismatch);
                }
            }
            ChoiceType::Boolean => {
                if let Constraints::Boolean(_) = constraints {
                    ChoiceValue::Boolean(false) // false is typically the simplest boolean
                } else {
                    return Err(TemplateError::ConstraintMismatch);
                }
            }
            ChoiceType::Float => {
                if let Constraints::Float(float_constraints) = constraints {
                    let target = 0.0f64;
                    let clamped = if float_constraints.min_value.is_finite() 
                        && float_constraints.max_value.is_finite() {
                        target
                            .max(float_constraints.min_value)
                            .min(float_constraints.max_value)
                    } else {
                        target
                    };
                    ChoiceValue::Float(clamped)
                } else {
                    return Err(TemplateError::ConstraintMismatch);
                }
            }
            ChoiceType::String => {
                if let Constraints::String(string_constraints) = constraints {
                    if string_constraints.min_size == 0 {
                        ChoiceValue::String(String::new())
                    } else {
                        // Generate minimal string that satisfies min_size
                        let minimal_char = string_constraints.intervals.intervals
                            .first()
                            .and_then(|(start, _)| char::from_u32(*start))
                            .unwrap_or('a');
                        let minimal_string = minimal_char
                            .to_string()
                            .repeat(string_constraints.min_size);
                        ChoiceValue::String(minimal_string)
                    }
                } else {
                    return Err(TemplateError::ConstraintMismatch);
                }
            }
            ChoiceType::Bytes => {
                if let Constraints::Bytes(bytes_constraints) = constraints {
                    if bytes_constraints.min_size == 0 {
                        ChoiceValue::Bytes(Vec::new())
                    } else {
                        // Generate minimal bytes that satisfy min_size
                        ChoiceValue::Bytes(vec![0u8; bytes_constraints.min_size])
                    }
                } else {
                    return Err(TemplateError::ConstraintMismatch);
                }
            }
        };
        
        Ok(ChoiceNode::new(
            choice_type,
            value,
            constraints.clone(),
            false, // Generated choices are not marked as forced by default
        ))
    }
    
    /// Generate a choice at a specific index using choice_from_index logic
    fn generate_choice_at_index(
        &self,
        index: usize,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceNode, TemplateError> {
        log::debug!("[TEMPLATE] Generating choice at index {} for type {:?}", index, choice_type);
        
        // Use the existing choice_from_index function from indexing module
        use crate::choice::indexing::choice_from_index;
        
        let choice_type_str = match choice_type {
            ChoiceType::Integer => "integer",
            ChoiceType::Boolean => "boolean", 
            ChoiceType::Float => "float",
            ChoiceType::String => "string",
            ChoiceType::Bytes => "bytes",
        };
        
        let value = choice_from_index(index as u128, choice_type_str, constraints);
        
        log::debug!("[TEMPLATE] Generated value {:?} from index {}", value, index);
        
        Ok(ChoiceNode::new(
            choice_type,
            value,
            constraints.clone(),
            false, // Index-based choices are not marked as forced
        ))
    }
    
    /// Generate a biased choice using weighted selection with specified bias
    fn generate_biased_choice(
        &self,
        bias: f64,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceNode, TemplateError> {
        log::debug!("[TEMPLATE] Generating biased choice with bias {} for type {:?}", bias, choice_type);
        
        // Clamp bias to valid range [0.0, 1.0]
        let clamped_bias = bias.max(0.0).min(1.0);
        
        // Apply bias-based choice generation strategy
        let value = match choice_type {
            ChoiceType::Boolean => {
                // For booleans, bias represents probability of True
                let bool_value = clamped_bias > 0.5;
                log::debug!("[TEMPLATE] Biased boolean: bias {} -> {}", clamped_bias, bool_value);
                ChoiceValue::Boolean(bool_value)
            },
            ChoiceType::Integer => {
                // For integers, bias determines proximity to shrink_towards
                if let Constraints::Integer(int_constraints) = constraints {
                    let shrink_towards = int_constraints.shrink_towards.unwrap_or(0);
                    
                    // Use bias to determine distance from shrink_towards
                    // bias close to 0.0 = closer to shrink_towards
                    // bias close to 1.0 = farther from shrink_towards
                    let max_distance = 100; // Reasonable distance for biased generation
                    let distance = (clamped_bias * max_distance as f64) as i128;
                    
                    // Alternate positive/negative based on bias fractional part
                    let direction = if (clamped_bias * 1000.0) as i64 % 2 == 0 { 1 } else { -1 };
                    let biased_value = shrink_towards + direction * distance;
                    
                    // Apply constraints
                    let final_value = match (int_constraints.min_value, int_constraints.max_value) {
                        (Some(min), Some(max)) => biased_value.max(min).min(max),
                        (Some(min), None) => biased_value.max(min),
                        (None, Some(max)) => biased_value.min(max),
                        (None, None) => biased_value,
                    };
                    
                    log::debug!("[TEMPLATE] Biased integer: shrink_towards={}, distance={}, direction={}, final={}", 
                              shrink_towards, distance, direction, final_value);
                    ChoiceValue::Integer(final_value)
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            ChoiceType::Float => {
                // For floats, use bias to determine magnitude and sign
                let base_magnitude = clamped_bias * 1000.0; // Scale bias to reasonable float range
                let sign = if clamped_bias < 0.5 { -1.0 } else { 1.0 };
                let biased_float = sign * base_magnitude;
                
                // Apply float constraints
                if let Constraints::Float(float_constraints) = constraints {
                    let clamped_float = biased_float
                        .max(float_constraints.min_value)
                        .min(float_constraints.max_value);
                    
                    log::debug!("[TEMPLATE] Biased float: bias {} -> magnitude={}, sign={}, final={}", 
                              clamped_bias, base_magnitude, sign, clamped_float);
                    ChoiceValue::Float(clamped_float)
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            ChoiceType::String => {
                // For strings, bias determines length preference
                let max_length = if let Constraints::String(str_constraints) = constraints {
                    str_constraints.max_size
                } else {
                    return Err(TemplateError::TypeMismatch);
                };
                
                let biased_length = (clamped_bias * max_length as f64) as usize;
                let biased_string = "a".repeat(biased_length); // Simple bias toward repeated 'a'
                
                log::debug!("[TEMPLATE] Biased string: bias {} -> length {}", clamped_bias, biased_length);
                ChoiceValue::String(biased_string)
            },
            ChoiceType::Bytes => {
                // For bytes, bias determines length and content preference
                let max_length = if let Constraints::Bytes(byte_constraints) = constraints {
                    byte_constraints.max_size
                } else {
                    return Err(TemplateError::TypeMismatch);
                };
                
                let biased_length = (clamped_bias * max_length as f64) as usize;
                let biased_byte = (clamped_bias * 255.0) as u8;
                let biased_bytes = vec![biased_byte; biased_length];
                
                log::debug!("[TEMPLATE] Biased bytes: bias {} -> length {}, byte value {}", 
                          clamped_bias, biased_length, biased_byte);
                ChoiceValue::Bytes(biased_bytes)
            },
        };
        
        Ok(ChoiceNode::new(
            choice_type,
            value,
            constraints.clone(),
            false, // Biased choices are not marked as forced
        ))
    }
    
    /// Generate a custom choice using named template strategies
    fn generate_custom_choice(
        &self,
        name: &str,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceNode, TemplateError> {
        log::debug!("[TEMPLATE] Generating custom choice '{}' for type {:?}", name, choice_type);
        
        // Implement named template strategies based on Python Hypothesis patterns
        let value = match (name, choice_type) {
            // Boundary value templates
            ("boundary_min", ChoiceType::Integer) => {
                if let Constraints::Integer(int_constraints) = constraints {
                    let min_val = int_constraints.min_value.unwrap_or(i128::MIN);
                    log::debug!("[TEMPLATE] Custom boundary_min integer: {}", min_val);
                    ChoiceValue::Integer(min_val)
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            ("boundary_max", ChoiceType::Integer) => {
                if let Constraints::Integer(int_constraints) = constraints {
                    let max_val = int_constraints.max_value.unwrap_or(i128::MAX);
                    log::debug!("[TEMPLATE] Custom boundary_max integer: {}", max_val);
                    ChoiceValue::Integer(max_val)
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            ("zero", ChoiceType::Integer) => {
                log::debug!("[TEMPLATE] Custom zero integer");
                ChoiceValue::Integer(0)
            },
            ("one", ChoiceType::Integer) => {
                log::debug!("[TEMPLATE] Custom one integer");
                ChoiceValue::Integer(1)
            },
            
            // Float boundary templates
            ("boundary_min", ChoiceType::Float) => {
                if let Constraints::Float(float_constraints) = constraints {
                    let min_val = float_constraints.min_value;
                    log::debug!("[TEMPLATE] Custom boundary_min float: {}", min_val);
                    ChoiceValue::Float(min_val)
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            ("boundary_max", ChoiceType::Float) => {
                if let Constraints::Float(float_constraints) = constraints {
                    let max_val = float_constraints.max_value;
                    log::debug!("[TEMPLATE] Custom boundary_max float: {}", max_val);
                    ChoiceValue::Float(max_val)
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            ("zero", ChoiceType::Float) => {
                log::debug!("[TEMPLATE] Custom zero float");
                ChoiceValue::Float(0.0)
            },
            ("one", ChoiceType::Float) => {
                log::debug!("[TEMPLATE] Custom one float");
                ChoiceValue::Float(1.0)
            },
            ("infinity", ChoiceType::Float) => {
                if let Constraints::Float(float_constraints) = constraints {
                    // Check if infinity is allowed by the constraints (infinity should be within the bounds)
                    let allow_infinity = float_constraints.max_value.is_infinite() && float_constraints.max_value > 0.0;
                    if allow_infinity {
                        log::debug!("[TEMPLATE] Custom infinity float");
                        ChoiceValue::Float(f64::INFINITY)
                    } else {
                        // Fall back to max value if infinity not allowed
                        log::debug!("[TEMPLATE] Custom infinity float (not allowed, using max): {}", float_constraints.max_value);
                        ChoiceValue::Float(float_constraints.max_value)
                    }
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            ("nan", ChoiceType::Float) => {
                if let Constraints::Float(float_constraints) = constraints {
                    if float_constraints.allow_nan {
                        log::debug!("[TEMPLATE] Custom NaN float");
                        ChoiceValue::Float(f64::NAN)
                    } else {
                        // Fall back to zero if NaN not allowed
                        log::debug!("[TEMPLATE] Custom NaN float (not allowed, using zero)");
                        ChoiceValue::Float(0.0)
                    }
                } else {
                    return Err(TemplateError::TypeMismatch);
                }
            },
            
            // Boolean templates
            ("true", ChoiceType::Boolean) => {
                log::debug!("[TEMPLATE] Custom true boolean");
                ChoiceValue::Boolean(true)
            },
            ("false", ChoiceType::Boolean) => {
                log::debug!("[TEMPLATE] Custom false boolean");
                ChoiceValue::Boolean(false)
            },
            
            // String templates
            ("empty", ChoiceType::String) => {
                log::debug!("[TEMPLATE] Custom empty string");
                ChoiceValue::String(String::new())
            },
            ("single_char", ChoiceType::String) => {
                log::debug!("[TEMPLATE] Custom single_char string");
                ChoiceValue::String("a".to_string())
            },
            ("whitespace", ChoiceType::String) => {
                log::debug!("[TEMPLATE] Custom whitespace string");
                ChoiceValue::String(" \t\n".to_string())
            },
            ("unicode", ChoiceType::String) => {
                log::debug!("[TEMPLATE] Custom unicode string");
                ChoiceValue::String("αβγδε".to_string()) // Greek letters for unicode testing
            },
            
            // Bytes templates
            ("empty", ChoiceType::Bytes) => {
                log::debug!("[TEMPLATE] Custom empty bytes");
                ChoiceValue::Bytes(Vec::new())
            },
            ("single_byte", ChoiceType::Bytes) => {
                log::debug!("[TEMPLATE] Custom single_byte bytes");
                ChoiceValue::Bytes(vec![0x42]) // 'B' in ASCII
            },
            ("null_bytes", ChoiceType::Bytes) => {
                log::debug!("[TEMPLATE] Custom null_bytes");
                ChoiceValue::Bytes(vec![0x00; 4]) // Four null bytes
            },
            ("high_bytes", ChoiceType::Bytes) => {
                log::debug!("[TEMPLATE] Custom high_bytes");
                ChoiceValue::Bytes(vec![0xFF, 0xFE, 0xFD, 0xFC]) // High byte values
            },
            
            // Fallback for unknown templates
            _ => {
                log::warn!("[TEMPLATE] Unknown custom template '{}' for type {:?}, falling back to simplest", name, choice_type);
                return self.generate_simplest_choice(choice_type, constraints);
            }
        };
        
        Ok(ChoiceNode::new(
            choice_type,
            value,
            constraints.clone(),
            false, // Custom choices are not marked as forced
        ))
    }
    
    /// Check if a value matches the expected choice type
    fn value_matches_type(&self, value: &ChoiceValue, choice_type: ChoiceType) -> bool {
        match (value, choice_type) {
            (ChoiceValue::Integer(_), ChoiceType::Integer) => true,
            (ChoiceValue::Boolean(_), ChoiceType::Boolean) => true,
            (ChoiceValue::Float(_), ChoiceType::Float) => true,
            (ChoiceValue::String(_), ChoiceType::String) => true,
            (ChoiceValue::Bytes(_), ChoiceType::Bytes) => true,
            _ => false,
        }
    }
    
    /// Check if a value satisfies the given constraints (simplified validation)
    fn value_satisfies_constraints(&self, value: &ChoiceValue, constraints: &Constraints) -> bool {
        match (value, constraints) {
            (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
                let min_ok = c.min_value.map_or(true, |min| *val >= min);
                let max_ok = c.max_value.map_or(true, |max| *val <= max);
                min_ok && max_ok
            }
            (ChoiceValue::Boolean(_), Constraints::Boolean(_)) => true,
            (ChoiceValue::Float(val), Constraints::Float(c)) => {
                if val.is_nan() {
                    c.allow_nan
                } else {
                    *val >= c.min_value && *val <= c.max_value
                }
            }
            (ChoiceValue::String(val), Constraints::String(c)) => {
                val.len() >= c.min_size && val.len() <= c.max_size
            }
            (ChoiceValue::Bytes(val), Constraints::Bytes(c)) => {
                val.len() >= c.min_size && val.len() <= c.max_size
            }
            _ => false,
        }
    }
    
    /// Record that a misalignment occurred at the current position
    fn record_misalignment(&mut self) {
        if self.misalignment_index.is_none() {
            self.misalignment_index = Some(self.processed_count);
        }
    }
    
    /// Fall back to generating the simplest valid choice
    fn fallback_to_simplest(
        &self,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<Option<ChoiceNode>, TemplateError> {
        let node = self.generate_simplest_choice(choice_type, constraints)?;
        Ok(Some(node))
    }
    
    /// Calculate expected choice count from template entries
    pub fn calculate_expected_choice_count(&self) -> usize {
        self.template_queue.iter()
            .map(|entry| match entry {
                TemplateEntry::Template(template) => {
                    match template.count {
                        Some(usize::MAX) => 1, // Treat unlimited as 1 for count calculations
                        Some(count) => count,
                        None => 1, // Single-use templates
                    }
                }
                _ => 1,
            })
            .sum::<usize>() + self.processed_count
    }
    
    /// Get debug information about the template engine state
    pub fn debug_info(&self) -> String {
        format!(
            "TemplateEngine {{ remaining: {}, processed: {}, misalignment: {:?}, debug: {} }}",
            self.remaining_count(),
            self.processed_count(),
            self.misalignment_index(),
            self.debug_mode
        )
    }
    
    /// Reset the template engine to initial state
    pub fn reset(&mut self) {
        self.template_queue.clear();
        self.processed_count = 0;
        self.misalignment_index = None;
    }
    
    /// Clone the current state for analysis or backup
    pub fn clone_state(&self) -> TemplateEngineState {
        TemplateEngineState {
            remaining_entries: self.template_queue.clone(),
            processed_count: self.processed_count,
            misalignment_index: self.misalignment_index,
        }
    }
    
    /// Restore from a previously saved state
    pub fn restore_state(&mut self, state: TemplateEngineState) {
        self.template_queue = state.remaining_entries;
        self.processed_count = state.processed_count;
        self.misalignment_index = state.misalignment_index;
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Saved state of a template engine for backup/restore operations
#[derive(Debug, Clone)]
pub struct TemplateEngineState {
    pub remaining_entries: VecDeque<TemplateEntry>,
    pub processed_count: usize,
    pub misalignment_index: Option<usize>,
}

/// Advanced template generation capabilities - Public API
impl TemplateEngine {
    /// Generate a choice at a specific index for deterministic test case construction
    /// 
    /// This method uses the choice indexing system to generate values at specific positions
    /// in the shrinking order, enabling precise control over test case generation.
    pub fn generate_at_index(
        &self,
        index: usize,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceNode, TemplateError> {
        log::info!("[TEMPLATE_API] Generating choice at index {} for type {:?}", index, choice_type);
        self.generate_choice_at_index(index, choice_type, constraints)
    }
    
    /// Generate a biased choice with the specified probability bias
    /// 
    /// The bias parameter (0.0 to 1.0) influences choice generation:
    /// - For booleans: bias is probability of True
    /// - For integers: bias determines distance from shrink_towards 
    /// - For floats: bias affects magnitude and sign
    /// - For strings/bytes: bias determines length preferences
    pub fn generate_with_bias(
        &self,
        bias: f64,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceNode, TemplateError> {
        log::info!("[TEMPLATE_API] Generating biased choice with bias {} for type {:?}", bias, choice_type);
        
        if bias < 0.0 || bias > 1.0 {
            return Err(TemplateError::ProcessingFailed(
                format!("Bias must be between 0.0 and 1.0, got {}", bias)
            ));
        }
        
        self.generate_biased_choice(bias, choice_type, constraints)
    }
    
    /// Generate a custom choice using named template strategies
    /// 
    /// Supported template names:
    /// - "boundary_min", "boundary_max": Boundary values for integers/floats
    /// - "zero", "one": Common numeric values
    /// - "infinity", "nan": Special float values (if allowed by constraints)
    /// - "true", "false": Boolean values
    /// - "empty", "single_char", "whitespace", "unicode": String patterns
    /// - "single_byte", "null_bytes", "high_bytes": Byte patterns
    pub fn generate_custom(
        &self,
        template_name: &str,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceNode, TemplateError> {
        log::info!("[TEMPLATE_API] Generating custom choice '{}' for type {:?}", template_name, choice_type);
        self.generate_custom_choice(template_name, choice_type, constraints)
    }
    
    /// Generate a series of choices at consecutive indices
    /// 
    /// This is useful for generating deterministic sequences of values
    /// starting from a specific index position.
    pub fn generate_sequence_from_index(
        &self,
        start_index: usize,
        count: usize,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<Vec<ChoiceNode>, TemplateError> {
        log::info!("[TEMPLATE_API] Generating sequence from index {} with {} choices for type {:?}", 
                  start_index, count, choice_type);
        
        let mut choices = Vec::with_capacity(count);
        for i in 0..count {
            let choice = self.generate_choice_at_index(start_index + i, choice_type, constraints)?;
            choices.push(choice);
        }
        
        log::debug!("[TEMPLATE_API] Generated {} choices in sequence", choices.len());
        Ok(choices)
    }
    
    /// Generate choices with graduated bias (useful for exploration)
    /// 
    /// Creates a series of choices with evenly distributed bias values
    /// from 0.0 to 1.0, enabling systematic exploration of the bias space.
    pub fn generate_biased_sequence(
        &self,
        count: usize,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<Vec<ChoiceNode>, TemplateError> {
        log::info!("[TEMPLATE_API] Generating biased sequence with {} choices for type {:?}", count, choice_type);
        
        if count == 0 {
            return Ok(Vec::new());
        }
        
        let mut choices = Vec::with_capacity(count);
        
        // Generate evenly distributed bias values from 0.0 to 1.0
        for i in 0..count {
            let bias = if count == 1 {
                0.5 // Single choice gets middle bias
            } else {
                i as f64 / (count - 1) as f64 // Evenly distribute from 0.0 to 1.0
            };
            
            let choice = self.generate_biased_choice(bias, choice_type, constraints)?;
            choices.push(choice);
        }
        
        log::debug!("[TEMPLATE_API] Generated {} biased choices", choices.len());
        Ok(choices)
    }
    
    /// Generate all available custom template choices for a given type
    /// 
    /// Returns a collection of choices generated using all applicable
    /// custom templates for the specified choice type and constraints.
    pub fn generate_all_custom_templates(
        &self,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<Vec<(String, ChoiceNode)>, TemplateError> {
        log::info!("[TEMPLATE_API] Generating all custom templates for type {:?}", choice_type);
        
        let template_names = match choice_type {
            ChoiceType::Integer => vec![
                "boundary_min", "boundary_max", "zero", "one"
            ],
            ChoiceType::Float => vec![
                "boundary_min", "boundary_max", "zero", "one", "infinity", "nan"
            ],
            ChoiceType::Boolean => vec![
                "true", "false"
            ],
            ChoiceType::String => vec![
                "empty", "single_char", "whitespace", "unicode"
            ],
            ChoiceType::Bytes => vec![
                "empty", "single_byte", "null_bytes", "high_bytes"
            ],
        };
        
        let mut results = Vec::new();
        
        for &template_name in &template_names {
            match self.generate_custom_choice(template_name, choice_type, constraints) {
                Ok(choice) => {
                    results.push((template_name.to_string(), choice));
                },
                Err(e) => {
                    log::warn!("[TEMPLATE_API] Failed to generate custom template '{}': {:?}", template_name, e);
                    // Continue with other templates rather than failing completely
                }
            }
        }
        
        log::debug!("[TEMPLATE_API] Generated {} custom template choices", results.len());
        Ok(results)
    }
}

/// Errors that can occur during template processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// Template has been exhausted (no remaining uses)
    ExhaustedTemplate,
    /// Template value doesn't match expected constraints
    ConstraintMismatch,
    /// Template type doesn't match expected choice type
    TypeMismatch,
    /// Custom template name not found or not implemented
    UnknownCustomTemplate(String),
    /// Template processing failed for unknown reason
    ProcessingFailed(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::ExhaustedTemplate => {
                write!(f, "Template has no remaining uses")
            }
            TemplateError::ConstraintMismatch => {
                write!(f, "Template value violates constraints")
            }
            TemplateError::TypeMismatch => {
                write!(f, "Template type doesn't match expected choice type")
            }
            TemplateError::UnknownCustomTemplate(name) => {
                write!(f, "Unknown custom template: {}", name)
            }
            TemplateError::ProcessingFailed(msg) => {
                write!(f, "Template processing failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for TemplateError {}

/// Convenience functions for creating common template patterns
pub mod templates {
    use super::*;
    
    /// Create a sequence of simplest templates
    pub fn simplest_sequence(count: usize) -> Vec<TemplateEntry> {
        (0..count)
            .map(|_| TemplateEntry::Template(ChoiceTemplate::simplest()))
            .collect()
    }
    
    /// Create a sequence from direct values
    pub fn value_sequence(values: Vec<ChoiceValue>) -> Vec<TemplateEntry> {
        values
            .into_iter()
            .map(TemplateEntry::DirectValue)
            .collect()
    }
    
    /// Create a mixed sequence of templates and values
    pub fn mixed_sequence(
        templates: Vec<ChoiceTemplate>,
        values: Vec<ChoiceValue>,
    ) -> Vec<TemplateEntry> {
        let mut entries = Vec::new();
        
        for template in templates {
            entries.push(TemplateEntry::Template(template));
        }
        
        for value in values {
            entries.push(TemplateEntry::DirectValue(value));
        }
        
        entries
    }
    
    /// Create an index-based template sequence
    pub fn index_sequence(indices: Vec<usize>) -> Vec<TemplateEntry> {
        indices
            .into_iter()
            .map(|idx| TemplateEntry::Template(ChoiceTemplate::at_index(idx)))
            .collect()
    }
    
    /// Create a biased template sequence
    pub fn biased_sequence(biases: Vec<f64>) -> Vec<TemplateEntry> {
        biases
            .into_iter()
            .map(|bias| TemplateEntry::Template(ChoiceTemplate::biased(bias)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints, FloatConstraints};
    
    #[test]
    fn test_choice_template_creation() {
        let template = ChoiceTemplate::simplest();
        assert_eq!(template.template_type, TemplateType::Simplest);
        assert!(template.is_forcing);
        assert_eq!(template.count, None);
        
        let indexed_template = ChoiceTemplate::at_index(5);
        if let TemplateType::AtIndex(idx) = indexed_template.template_type {
            assert_eq!(idx, 5);
        } else {
            panic!("Expected AtIndex template type");
        }
        
        let biased_template = ChoiceTemplate::biased(0.7);
        if let TemplateType::Biased { bias } = biased_template.template_type {
            assert!((bias - 0.7).abs() < f64::EPSILON);
        } else {
            panic!("Expected Biased template type");
        }
        assert!(!biased_template.is_forcing); // Biased templates are not forcing
    }
    
    #[test]
    fn test_template_usage_counting() {
        let mut template = ChoiceTemplate::with_count(TemplateType::Simplest, 3);
        
        assert!(template.has_remaining_count());
        assert_eq!(template.remaining_count(), Some(3));
        
        template = template.consume_usage().unwrap();
        assert_eq!(template.remaining_count(), Some(2));
        
        template = template.consume_usage().unwrap();
        template = template.consume_usage().unwrap();
        assert_eq!(template.remaining_count(), Some(0));
        assert!(!template.has_remaining_count());
        
        let result = template.consume_usage();
        assert!(result.is_err());
    }
    
    #[test]
    fn test_template_entry_types() {
        let direct_entry = TemplateEntry::direct(ChoiceValue::Integer(42));
        assert!(direct_entry.is_forcing());
        
        let template_entry = TemplateEntry::template(ChoiceTemplate::biased(0.5));
        assert!(!template_entry.is_forcing()); // Biased templates are not forcing
        
        let partial_entry = TemplateEntry::partial_node(
            ChoiceValue::String("test".to_string()),
            None,
            Some("test partial node".to_string()),
        );
        assert!(partial_entry.is_forcing());
    }
    
    #[test]
    fn test_template_engine_creation() {
        let engine = TemplateEngine::new();
        assert!(!engine.has_templates());
        assert_eq!(engine.remaining_count(), 0);
        assert_eq!(engine.processed_count(), 0);
        
        let values = vec![
            ChoiceValue::Integer(1),
            ChoiceValue::Boolean(true),
            ChoiceValue::Float(3.14),
        ];
        let engine_with_values = TemplateEngine::from_values(values);
        assert!(engine_with_values.has_templates());
        assert_eq!(engine_with_values.remaining_count(), 3);
    }
    
    #[test]
    fn test_template_engine_processing() {
        let mut engine = TemplateEngine::from_values(vec![
            ChoiceValue::Integer(42),
            ChoiceValue::Boolean(true),
        ]);
        
        // Process integer template
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });
        
        let result = engine.process_next_template(
            ChoiceType::Integer,
            &int_constraints,
        ).unwrap();
        
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(node.was_forced);
        assert_eq!(engine.processed_count(), 1);
        
        // Process boolean template
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        
        let result = engine.process_next_template(
            ChoiceType::Boolean,
            &bool_constraints,
        ).unwrap();
        
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Boolean(true));
        assert!(node.was_forced);
        assert_eq!(engine.processed_count(), 2);
        
        // No more templates
        let result = engine.process_next_template(
            ChoiceType::Integer,
            &int_constraints,
        ).unwrap();
        assert!(result.is_none());
    }
    
    #[test]
    fn test_template_misalignment_handling() {
        let mut engine = TemplateEngine::from_values(vec![
            ChoiceValue::Integer(200), // Outside constraint range
        ]);
        
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });
        
        let result = engine.process_next_template(
            ChoiceType::Integer,
            &int_constraints,
        ).unwrap();
        
        assert!(result.is_some());
        let node = result.unwrap();
        // Should fall back to simplest choice (0, which is shrink_towards)
        assert_eq!(node.value, ChoiceValue::Integer(0));
        assert!(engine.has_misalignment());
        assert_eq!(engine.misalignment_index(), Some(1));
    }
    
    #[test]
    fn test_simplest_choice_generation() {
        let engine = TemplateEngine::new();
        
        // Test integer simplest
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(5),
            max_value: Some(15),
            weights: None,
            shrink_towards: Some(7),
        });
        
        let node = engine.generate_simplest_choice(
            ChoiceType::Integer,
            &int_constraints,
        ).unwrap();
        
        assert_eq!(node.value, ChoiceValue::Integer(7)); // shrink_towards value
        
        // Test boolean simplest
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        
        let node = engine.generate_simplest_choice(
            ChoiceType::Boolean,
            &bool_constraints,
        ).unwrap();
        
        assert_eq!(node.value, ChoiceValue::Boolean(false));
        
        // Test float simplest
        let float_constraints = Constraints::Float(FloatConstraints {
            min_value: -1.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        });
        
        let node = engine.generate_simplest_choice(
            ChoiceType::Float,
            &float_constraints,
        ).unwrap();
        
        assert_eq!(node.value, ChoiceValue::Float(0.0));
    }
    
    #[test]
    fn test_template_engine_state_management() {
        let mut engine = TemplateEngine::from_values(vec![
            ChoiceValue::Integer(1),
            ChoiceValue::Integer(2),
        ]);
        
        let state = engine.clone_state();
        assert_eq!(state.remaining_entries.len(), 2);
        assert_eq!(state.processed_count, 0);
        
        // Process one template
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let _ = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        
        assert_eq!(engine.remaining_count(), 1);
        assert_eq!(engine.processed_count(), 1);
        
        // Restore original state
        engine.restore_state(state);
        assert_eq!(engine.remaining_count(), 2);
        assert_eq!(engine.processed_count(), 0);
    }
    
    #[test]
    fn test_template_convenience_functions() {
        let simplest_seq = templates::simplest_sequence(3);
        assert_eq!(simplest_seq.len(), 3);
        for entry in &simplest_seq {
            if let TemplateEntry::Template(template) = entry {
                assert_eq!(template.template_type, TemplateType::Simplest);
            } else {
                panic!("Expected template entry");
            }
        }
        
        let values = vec![
            ChoiceValue::Integer(1),
            ChoiceValue::Boolean(true),
        ];
        let value_seq = templates::value_sequence(values.clone());
        assert_eq!(value_seq.len(), 2);
        
        let index_seq = templates::index_sequence(vec![0, 1, 2]);
        assert_eq!(index_seq.len(), 3);
        
        let biased_seq = templates::biased_sequence(vec![0.3, 0.7]);
        assert_eq!(biased_seq.len(), 2);
    }
    
    #[test]
    fn test_template_type_display() {
        assert_eq!(TemplateType::Simplest.to_string(), "simplest");
        assert_eq!(TemplateType::AtIndex(5).to_string(), "at_index(5)");
        assert_eq!(TemplateType::Biased { bias: 0.75 }.to_string(), "biased(0.750)");
        assert_eq!(
            TemplateType::Custom { name: "test".to_string() }.to_string(),
            "custom(test)"
        );
    }
    
    #[test]
    fn test_template_error_display() {
        assert_eq!(
            TemplateError::ExhaustedTemplate.to_string(),
            "Template has no remaining uses"
        );
        assert_eq!(
            TemplateError::ConstraintMismatch.to_string(),
            "Template value violates constraints"
        );
        assert_eq!(
            TemplateError::UnknownCustomTemplate("custom".to_string()).to_string(),
            "Unknown custom template: custom"
        );
    }
}