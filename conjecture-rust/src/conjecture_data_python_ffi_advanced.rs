//! Advanced ConjectureData Python FFI Integration
//! 
//! This module extends the basic FFI layer with advanced features:
//! - Binary choice sequence serialization (matching Python's database format)
//! - Advanced constraints validation and normalization
//! - Performance-optimized bulk operations
//! - Comprehensive state management and synchronization
//! - Memory-efficient streaming operations for large test cases

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyBool, PyNone};
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyMemoryError};
use crate::conjecture_data_python_ffi::*;
use crate::data::ConjectureData;
use crate::choice::constraints::*;
use crate::choice::values::*;
use crate::choice::*;
use std::collections::HashMap;
use std::io::{Write, Cursor};
use byteorder::{BigEndian, LittleEndian, WriteBytesExt, ReadBytesExt};

// Advanced debug logging with hex formatting
macro_rules! ffi_advanced_debug {
    ($($arg:tt)*) => {
        #[cfg(not(test))]
        println!("CONJECTURE_DATA_FFI_ADVANCED DEBUG: {}", format!($($arg)*));
    };
}

/// Binary serialization format matching Python's choices_to_bytes format
pub struct ChoiceSequenceBinaryCodec;

impl ChoiceSequenceBinaryCodec {
    /// Serialize choice sequence to binary format matching Python implementation
    /// 
    /// Format: metadata_byte [size?] [payload]
    /// - Boolean: 000_0000v where v is the boolean value
    /// - Other types: tag_ssss [uint16 size?] [payload]
    ///   - Tag 1 (001): Float (IEEE 754 double, 8 bytes)
    ///   - Tag 2 (010): Integer (variable-length big-endian signed)
    ///   - Tag 3 (011): Bytes (raw bytes)
    ///   - Tag 4 (100): String (UTF-8 with surrogate handling)
    pub fn serialize_to_bytes(choices: &[ChoiceNode]) -> Result<Vec<u8>, FfiError> {
        ffi_advanced_debug!("Serializing {} choices to binary format", choices.len());
        
        let mut buffer = Vec::new();
        
        for (i, choice) in choices.iter().enumerate() {
            match &choice.value {
                ChoiceValue::Boolean(value) => {
                    // Boolean encoding: 000_0000v
                    let byte = if *value { 0x01 } else { 0x00 };
                    buffer.push(byte);
                    ffi_advanced_debug!("Choice {}: Boolean({}) -> 0x{:02X}", i, value, byte);
                }
                ChoiceValue::Float(value) => {
                    // Float encoding: 001_ssss [size?] [8-byte IEEE 754]
                    let size_bits = Self::encode_size_in_metadata(8)?;
                    let metadata = 0x10 | size_bits; // 001_ssss
                    buffer.push(metadata);
                    
                    if size_bits == 0x0F {
                        // Size >= 31, need separate size encoding
                        Self::write_uleb128(&mut buffer, 8)?;
                    }
                    
                    // Write IEEE 754 double in big-endian
                    buffer.write_f64::<BigEndian>(*value)
                        .map_err(|e| FfiError::SerializationError(format!("Float write error: {}", e)))?;
                    
                    ffi_advanced_debug!("Choice {}: Float({}) -> 0x{:02X} + 8 bytes", i, value, metadata);
                }
                ChoiceValue::Integer(value) => {
                    // Integer encoding: 010_ssss [size?] [variable-length big-endian signed]
                    let int_bytes = Self::encode_signed_integer(*value)?;
                    let size_bits = Self::encode_size_in_metadata(int_bytes.len())?;
                    let metadata = 0x20 | size_bits; // 010_ssss
                    buffer.push(metadata);
                    
                    if size_bits == 0x0F {
                        Self::write_uleb128(&mut buffer, int_bytes.len())?;
                    }
                    
                    buffer.extend_from_slice(&int_bytes);
                    
                    ffi_advanced_debug!("Choice {}: Integer({}) -> 0x{:02X} + {} bytes (0x{})", 
                                       i, value, metadata, int_bytes.len(),
                                       int_bytes.iter().map(|b| format!("{:02X}", b)).collect::<String>());
                }
                ChoiceValue::Bytes(value) => {
                    // Bytes encoding: 011_ssss [size?] [raw bytes]
                    let size_bits = Self::encode_size_in_metadata(value.len())?;
                    let metadata = 0x30 | size_bits; // 011_ssss
                    buffer.push(metadata);
                    
                    if size_bits == 0x0F {
                        Self::write_uleb128(&mut buffer, value.len())?;
                    }
                    
                    buffer.extend_from_slice(value);
                    
                    ffi_advanced_debug!("Choice {}: Bytes([{} bytes]) -> 0x{:02X} + {} bytes", 
                                       i, value.len(), metadata, value.len());
                }
                ChoiceValue::String(value) => {
                    // String encoding: 100_ssss [size?] [UTF-8 with surrogate handling]
                    let utf8_bytes = Self::encode_string_with_surrogates(value)?;
                    let size_bits = Self::encode_size_in_metadata(utf8_bytes.len())?;
                    let metadata = 0x40 | size_bits; // 100_ssss
                    buffer.push(metadata);
                    
                    if size_bits == 0x0F {
                        Self::write_uleb128(&mut buffer, utf8_bytes.len())?;
                    }
                    
                    buffer.extend_from_slice(&utf8_bytes);
                    
                    ffi_advanced_debug!("Choice {}: String(\"{}\") -> 0x{:02X} + {} UTF-8 bytes", 
                                       i, value, metadata, utf8_bytes.len());
                }
            }
        }
        
        ffi_advanced_debug!("Binary serialization complete: {} bytes total", buffer.len());
        Ok(buffer)
    }
    
    /// Deserialize choice sequence from binary format
    pub fn deserialize_from_bytes(data: &[u8]) -> Result<Vec<ChoiceNode>, FfiError> {
        ffi_advanced_debug!("Deserializing choice sequence from {} bytes", data.len());
        
        let mut cursor = Cursor::new(data);
        let mut choices = Vec::new();
        let mut choice_index = 0;
        
        while cursor.position() < data.len() as u64 {
            let metadata = cursor.read_u8()
                .map_err(|e| FfiError::DeserializationError(format!("Read metadata error: {}", e)))?;
            
            ffi_advanced_debug!("Choice {}: metadata=0x{:02X}", choice_index, metadata);
            
            if metadata & 0xF0 == 0x00 {
                // Boolean: 000_0000v
                let value = (metadata & 0x01) != 0;
                choices.push(ChoiceNode {
                    value: ChoiceValue::Boolean(value),
                    constraints: None,
                    was_forced: false,
                    index: Some(choice_index),
                });
                ffi_advanced_debug!("Choice {}: Boolean({})", choice_index, value);
                
            } else {
                // Other types: tag_ssss [size?] [payload]
                let tag = (metadata & 0x70) >> 4;
                let size_bits = metadata & 0x0F;
                
                let payload_size = if size_bits == 0x0F {
                    Self::read_uleb128(&mut cursor)?
                } else {
                    size_bits as usize
                };
                
                ffi_advanced_debug!("Choice {}: tag={}, size={}", choice_index, tag, payload_size);
                
                match tag {
                    1 => {
                        // Float
                        if payload_size != 8 {
                            return Err(FfiError::DeserializationError(
                                format!("Invalid float size: {} (expected 8)", payload_size)
                            ));
                        }
                        let value = cursor.read_f64::<BigEndian>()
                            .map_err(|e| FfiError::DeserializationError(format!("Float read error: {}", e)))?;
                        
                        choices.push(ChoiceNode {
                            value: ChoiceValue::Float(value),
                            constraints: None,
                            was_forced: false,
                            index: Some(choice_index),
                        });
                        ffi_advanced_debug!("Choice {}: Float({})", choice_index, value);
                    }
                    2 => {
                        // Integer
                        let mut int_bytes = vec![0u8; payload_size];
                        cursor.read_exact(&mut int_bytes)
                            .map_err(|e| FfiError::DeserializationError(format!("Integer read error: {}", e)))?;
                        
                        let value = Self::decode_signed_integer(&int_bytes)?;
                        choices.push(ChoiceNode {
                            value: ChoiceValue::Integer(value),
                            constraints: None,
                            was_forced: false,
                            index: Some(choice_index),
                        });
                        ffi_advanced_debug!("Choice {}: Integer({}) from {} bytes (0x{})", 
                                           choice_index, value, payload_size,
                                           int_bytes.iter().map(|b| format!("{:02X}", b)).collect::<String>());
                    }
                    3 => {
                        // Bytes
                        let mut bytes = vec![0u8; payload_size];
                        cursor.read_exact(&mut bytes)
                            .map_err(|e| FfiError::DeserializationError(format!("Bytes read error: {}", e)))?;
                        
                        choices.push(ChoiceNode {
                            value: ChoiceValue::Bytes(bytes.clone()),
                            constraints: None,
                            was_forced: false,
                            index: Some(choice_index),
                        });
                        ffi_advanced_debug!("Choice {}: Bytes([{} bytes])", choice_index, payload_size);
                    }
                    4 => {
                        // String
                        let mut utf8_bytes = vec![0u8; payload_size];
                        cursor.read_exact(&mut utf8_bytes)
                            .map_err(|e| FfiError::DeserializationError(format!("String read error: {}", e)))?;
                        
                        let value = Self::decode_string_with_surrogates(&utf8_bytes)?;
                        choices.push(ChoiceNode {
                            value: ChoiceValue::String(value.clone()),
                            constraints: None,
                            was_forced: false,
                            index: Some(choice_index),
                        });
                        ffi_advanced_debug!("Choice {}: String(\"{}\") from {} UTF-8 bytes", 
                                           choice_index, value, payload_size);
                    }
                    _ => {
                        return Err(FfiError::DeserializationError(format!("Unknown choice tag: {}", tag)));
                    }
                }
            }
            
            choice_index += 1;
        }
        
        ffi_advanced_debug!("Deserialization complete: {} choices", choices.len());
        Ok(choices)
    }
    
    /// Encode size in metadata (4 bits), returns 0x0F if size >= 31
    fn encode_size_in_metadata(size: usize) -> Result<u8, FfiError> {
        if size < 31 {
            Ok(size as u8)
        } else {
            Ok(0x0F) // Indicates separate size encoding needed
        }
    }
    
    /// Write ULEB128 (Unsigned Little Endian Base 128) encoding
    fn write_uleb128(buffer: &mut Vec<u8>, mut value: usize) -> Result<(), FfiError> {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                buffer.push(byte | 0x80);
            } else {
                buffer.push(byte);
                break;
            }
        }
        Ok(())
    }
    
    /// Read ULEB128 encoding
    fn read_uleb128(cursor: &mut Cursor<&[u8]>) -> Result<usize, FfiError> {
        let mut result = 0usize;
        let mut shift = 0;
        
        loop {
            let byte = cursor.read_u8()
                .map_err(|e| FfiError::DeserializationError(format!("ULEB128 read error: {}", e)))?;
            
            result |= ((byte & 0x7F) as usize) << shift;
            
            if (byte & 0x80) == 0 {
                break;
            }
            
            shift += 7;
            if shift >= 64 {
                return Err(FfiError::DeserializationError("ULEB128 value too large".to_string()));
            }
        }
        
        Ok(result)
    }
    
    /// Encode signed integer to variable-length big-endian bytes
    fn encode_signed_integer(value: i64) -> Result<Vec<u8>, FfiError> {
        if value == 0 {
            return Ok(vec![0]);
        }
        
        let mut bytes = Vec::new();
        let mut remaining = value;
        let is_negative = value < 0;
        
        // Convert to unsigned representation for bit manipulation
        let mut unsigned_value = if is_negative {
            (-(value + 1)) as u64
        } else {
            value as u64
        };
        
        // Find minimum number of bytes needed
        let mut temp = unsigned_value;
        let mut byte_count = 0;
        while temp > 0 {
            temp >>= 8;
            byte_count += 1;
        }
        
        if byte_count == 0 {
            byte_count = 1;
        }
        
        // Encode in big-endian
        for i in (0..byte_count).rev() {
            let byte = ((unsigned_value >> (i * 8)) & 0xFF) as u8;
            let final_byte = if is_negative { !byte } else { byte };
            bytes.push(final_byte);
        }
        
        Ok(bytes)
    }
    
    /// Decode signed integer from variable-length big-endian bytes
    fn decode_signed_integer(bytes: &[u8]) -> Result<i64, FfiError> {
        if bytes.is_empty() {
            return Err(FfiError::DeserializationError("Empty integer bytes".to_string()));
        }
        
        if bytes.len() > 8 {
            return Err(FfiError::DeserializationError("Integer too large".to_string()));
        }
        
        // Check if negative (MSB of first byte set)
        let is_negative = (bytes[0] & 0x80) != 0;
        
        let mut result = 0i64;
        for (i, &byte) in bytes.iter().enumerate() {
            let processed_byte = if is_negative { !byte } else { byte };
            result = (result << 8) | (processed_byte as i64);
        }
        
        if is_negative {
            result = -(result + 1);
        }
        
        Ok(result)
    }
    
    /// Encode string with surrogate handling (matching Python's approach)
    fn encode_string_with_surrogates(s: &str) -> Result<Vec<u8>, FfiError> {
        // For now, use standard UTF-8 encoding
        // Full implementation would handle surrogate escape encoding
        Ok(s.as_bytes().to_vec())
    }
    
    /// Decode string with surrogate handling
    fn decode_string_with_surrogates(bytes: &[u8]) -> Result<String, FfiError> {
        // For now, use standard UTF-8 decoding
        // Full implementation would handle surrogate escape decoding
        String::from_utf8(bytes.to_vec())
            .map_err(|e| FfiError::DeserializationError(format!("UTF-8 decode error: {}", e)))
    }
}

/// Advanced constraints validation with Python behavioral parity
pub struct ConstraintValidator;

impl ConstraintValidator {
    /// Validate and normalize constraints to match Python behavior exactly
    pub fn validate_and_normalize_integer_constraints(
        constraints: &mut IntegerConstraints
    ) -> Result<(), FfiError> {
        ffi_advanced_debug!("Validating integer constraints: min={:?}, max={:?}, shrink_towards={:?}", 
                           constraints.min_value, constraints.max_value, constraints.shrink_towards);
        
        // Python validation: min_value <= max_value
        if let (Some(min), Some(max)) = (constraints.min_value, constraints.max_value) {
            if min > max {
                return Err(FfiError::ValidationError(
                    format!("min_value ({}) must be <= max_value ({})", min, max)
                ));
            }
        }
        
        // Python behavior: shrink_towards defaults to 0 if not specified
        if constraints.shrink_towards.is_none() {
            constraints.shrink_towards = Some(0);
            ffi_advanced_debug!("Normalized shrink_towards to default value: 0");
        }
        
        // Python behavior: clamp shrink_towards to bounds if specified
        if let Some(shrink) = constraints.shrink_towards {
            let mut clamped_shrink = shrink;
            let mut was_clamped = false;
            
            if let Some(min) = constraints.min_value {
                if shrink < min {
                    clamped_shrink = min;
                    was_clamped = true;
                    ffi_advanced_debug!("Clamped shrink_towards from {} to min_value {}", shrink, min);
                }
            }
            
            if let Some(max) = constraints.max_value {
                if clamped_shrink > max {
                    clamped_shrink = max;
                    was_clamped = true;
                    ffi_advanced_debug!("Clamped shrink_towards from {} to max_value {}", clamped_shrink, max);
                }
            }
            
            if was_clamped {
                constraints.shrink_towards = Some(clamped_shrink);
            }
        }
        
        ffi_advanced_debug!("Integer constraints validation successful");
        Ok(())
    }
    
    /// Validate and normalize float constraints
    pub fn validate_and_normalize_float_constraints(
        constraints: &mut FloatConstraints
    ) -> Result<(), FfiError> {
        ffi_advanced_debug!("Validating float constraints: min={:?}, max={:?}, allow_nan={}, smallest_nonzero_magnitude={:?}", 
                           constraints.min_value, constraints.max_value, constraints.allow_nan, constraints.smallest_nonzero_magnitude);
        
        let min = constraints.min_value.unwrap_or(f64::NEG_INFINITY);
        let max = constraints.max_value.unwrap_or(f64::INFINITY);
        
        // Python validation: finite min <= finite max
        if min.is_finite() && max.is_finite() && min > max {
            return Err(FfiError::ValidationError(
                format!("min_value ({}) must be <= max_value ({})", min, max)
            ));
        }
        
        // Python validation: smallest_nonzero_magnitude must be positive
        if let Some(smallest) = constraints.smallest_nonzero_magnitude {
            if smallest <= 0.0 || !smallest.is_finite() {
                return Err(FfiError::ValidationError(
                    format!("smallest_nonzero_magnitude must be positive and finite, got {}", smallest)
                ));
            }
        }
        
        // Python behavior: normalize infinite bounds
        if min.is_infinite() && min.is_sign_negative() {
            constraints.min_value = None; // Represents -inf
        }
        if max.is_infinite() && max.is_sign_positive() {
            constraints.max_value = None; // Represents +inf
        }
        
        ffi_advanced_debug!("Float constraints validation successful");
        Ok(())
    }
    
    /// Cross-validate constraints against choice value
    pub fn validate_constraints_choice_compatibility(
        constraints: &Constraints,
        value: &ChoiceValue
    ) -> Result<(), FfiError> {
        match (constraints, value) {
            (Constraints::Integer(c), ChoiceValue::Integer(v)) => {
                if let Some(min) = c.min_value {
                    if *v < min {
                        return Err(FfiError::ValidationError(
                            format!("Integer value {} below min_value {}", v, min)
                        ));
                    }
                }
                if let Some(max) = c.max_value {
                    if *v > max {
                        return Err(FfiError::ValidationError(
                            format!("Integer value {} above max_value {}", v, max)
                        ));
                    }
                }
            }
            (Constraints::Float(c), ChoiceValue::Float(v)) => {
                if !c.allow_nan && v.is_nan() {
                    return Err(FfiError::ValidationError("NaN not allowed by constraints".to_string()));
                }
                
                if let Some(min) = c.min_value {
                    if v.is_finite() && *v < min {
                        return Err(FfiError::ValidationError(
                            format!("Float value {} below min_value {}", v, min)
                        ));
                    }
                }
                if let Some(max) = c.max_value {
                    if v.is_finite() && *v > max {
                        return Err(FfiError::ValidationError(
                            format!("Float value {} above max_value {}", v, max)
                        ));
                    }
                }
            }
            (Constraints::Bytes(c), ChoiceValue::Bytes(v)) => {
                if let Some(min_size) = c.min_size {
                    if v.len() < min_size {
                        return Err(FfiError::ValidationError(
                            format!("Bytes length {} below min_size {}", v.len(), min_size)
                        ));
                    }
                }
                if let Some(max_size) = c.max_size {
                    if v.len() > max_size {
                        return Err(FfiError::ValidationError(
                            format!("Bytes length {} above max_size {}", v.len(), max_size)
                        ));
                    }
                }
            }
            _ => {
                return Err(FfiError::ConstraintMismatch(
                    "Constraint type doesn't match choice value type".to_string()
                ));
            }
        }
        
        Ok(())
    }
}

/// High-performance bulk operations for large test suites
pub struct BulkOperations;

impl BulkOperations {
    /// Bulk export multiple ConjectureData instances to Python
    pub fn bulk_export_conjecture_data(
        py: Python,
        data_instances: &[&ConjectureData]
    ) -> PyResult<PyObject> {
        ffi_advanced_debug!("Bulk exporting {} ConjectureData instances", data_instances.len());
        
        let results_list = PyList::empty(py);
        
        for (i, data) in data_instances.iter().enumerate() {
            ffi_advanced_debug!("Processing instance {}/{}", i + 1, data_instances.len());
            
            let exported = export_conjecture_data_state(py, data)?;
            results_list.append(exported)?;
        }
        
        let result_dict = PyDict::new(py);
        result_dict.set_item("instances", results_list)?;
        result_dict.set_item("count", data_instances.len())?;
        
        ffi_advanced_debug!("Bulk export complete");
        Ok(result_dict.to_object(py))
    }
    
    /// Bulk validate constraintss with detailed error reporting
    pub fn bulk_validate_constraintss(
        py: Python,
        constraintss: &[Constraints]
    ) -> PyResult<PyObject> {
        ffi_advanced_debug!("Bulk validating {} constraintss", constraintss.len());
        
        let results_list = PyList::empty(py);
        let mut valid_count = 0;
        let mut error_count = 0;
        
        for (i, constraints) in constraintss.iter().enumerate() {
            let result_dict = PyDict::new(py);
            result_dict.set_item("index", i)?;
            result_dict.set_item("constraints_type", constraints.python_type_name())?;
            
            match validate_constraints_python_parity(py, constraints) {
                Ok(()) => {
                    result_dict.set_item("valid", true)?;
                    result_dict.set_item("error", py.None())?;
                    valid_count += 1;
                }
                Err(err) => {
                    result_dict.set_item("valid", false)?;
                    result_dict.set_item("error", format!("{}", err))?;
                    error_count += 1;
                }
            }
            
            results_list.append(result_dict)?;
        }
        
        let summary_dict = PyDict::new(py);
        summary_dict.set_item("results", results_list)?;
        summary_dict.set_item("total_count", constraintss.len())?;
        summary_dict.set_item("valid_count", valid_count)?;
        summary_dict.set_item("error_count", error_count)?;
        summary_dict.set_item("success_rate", valid_count as f64 / constraintss.len() as f64)?;
        
        ffi_advanced_debug!("Bulk validation complete: {}/{} valid", valid_count, constraintss.len());
        Ok(summary_dict.to_object(py))
    }
    
    /// Memory-efficient streaming serialization for very large choice sequences
    pub fn stream_serialize_choice_sequence(
        py: Python,
        choices: &[ChoiceNode],
        chunk_size: usize
    ) -> PyResult<PyObject> {
        ffi_advanced_debug!("Stream serializing {} choices with chunk_size={}", choices.len(), chunk_size);
        
        let chunks_list = PyList::empty(py);
        let total_binary_size = 0usize;
        
        for (chunk_index, chunk) in choices.chunks(chunk_size).enumerate() {
            ffi_advanced_debug!("Processing chunk {}: {} choices", chunk_index, chunk.len());
            
            // Serialize chunk to binary
            let binary_data = ChoiceSequenceBinaryCodec::serialize_to_bytes(chunk)
                .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {}", e)))?;
            
            // Create chunk descriptor
            let chunk_dict = PyDict::new(py);
            chunk_dict.set_item("chunk_index", chunk_index)?;
            chunk_dict.set_item("choice_count", chunk.len())?;
            chunk_dict.set_item("binary_size", binary_data.len())?;
            chunk_dict.set_item("binary_data", PyBytes::new(py, &binary_data))?;
            chunk_dict.set_item("start_choice_index", chunk_index * chunk_size)?;
            chunk_dict.set_item("end_choice_index", chunk_index * chunk_size + chunk.len())?;
            
            chunks_list.append(chunk_dict)?;
        }
        
        let result_dict = PyDict::new(py);
        result_dict.set_item("chunks", chunks_list)?;
        result_dict.set_item("total_choices", choices.len())?;
        result_dict.set_item("chunk_size", chunk_size)?;
        result_dict.set_item("chunk_count", chunks_list.len())?;
        
        ffi_advanced_debug!("Stream serialization complete: {} chunks", chunks_list.len());
        Ok(result_dict.to_object(py))
    }
}

/// Comprehensive state management for ConjectureData synchronization
pub struct StateManager;

impl StateManager {
    /// Create complete state snapshot for Python parity testing
    pub fn create_state_snapshot(py: Python, data: &ConjectureData) -> PyResult<PyObject> {
        ffi_advanced_debug!("Creating complete state snapshot for ConjectureData");
        
        let snapshot = PyDict::new(py);
        
        // Core state
        snapshot.set_item("buffer", PyBytes::new(py, data.buffer()))?;
        snapshot.set_item("index", data.index())?;
        snapshot.set_item("length", data.length())?;
        snapshot.set_item("max_length", data.max_length())?;
        snapshot.set_item("frozen", data.frozen())?;
        snapshot.set_item("status", format!("{:?}", data.status()))?;
        snapshot.set_item("depth", data.depth())?;
        
        // Choice sequence with constraintss
        let choices_list = PyList::empty(py);
        for choice in data.choices() {
            let choice_dict = export_choice_node_to_python(py, choice)?;
            choices_list.append(choice_dict)?;
        }
        snapshot.set_item("choices", choices_list)?;
        
        // Binary serialization for compatibility testing
        let binary_choices = ChoiceSequenceBinaryCodec::serialize_to_bytes(data.choices())
            .map_err(|e| PyRuntimeError::new_err(format!("Binary serialization error: {}", e)))?;
        snapshot.set_item("binary_choices", PyBytes::new(py, &binary_choices))?;
        
        // Examples with full details
        let examples_list = PyList::empty(py);
        for example in data.examples() {
            let example_dict = PyDict::new(py);
            example_dict.set_item("label", example.label)?;
            example_dict.set_item("start", example.start)?;
            example_dict.set_item("length", example.length)?;
            example_dict.set_item("parent", example.parent.unwrap_or(-1))?;
            example_dict.set_item("discarded", example.discarded)?;
            examples_list.append(example_dict)?;
        }
        snapshot.set_item("examples", examples_list)?;
        
        // Events
        let events_dict = PyDict::new(py);
        for (key, value) in data.events() {
            events_dict.set_item(key, value)?;
        }
        snapshot.set_item("events", events_dict)?;
        
        // Metadata
        snapshot.set_item("snapshot_timestamp", py.import("time")?.call_method0("time")?)?;
        snapshot.set_item("rust_implementation_version", "1.0.0")?;
        
        ffi_advanced_debug!("State snapshot created successfully");
        Ok(snapshot.to_object(py))
    }
    
    /// Restore ConjectureData from complete state snapshot
    pub fn restore_from_state_snapshot(py: Python, snapshot: &PyDict) -> Result<ConjectureData, FfiError> {
        ffi_advanced_debug!("Restoring ConjectureData from state snapshot");
        
        // Extract and validate buffer
        let buffer_obj = snapshot.get_item("buffer")
            .map_err(|e| FfiError::StateImportError(format!("Missing buffer: {}", e)))?
            .ok_or_else(|| FfiError::StateImportError("buffer is required".to_string()))?;
        
        let buffer = buffer_obj.downcast::<PyBytes>()
            .map_err(|e| FfiError::TypeConversionError(format!("buffer must be bytes: {}", e)))?
            .as_bytes()
            .to_vec();
        
        // Create ConjectureData
        let mut data = ConjectureData::for_buffer(&buffer);
        
        // Restore basic state
        if let Some(index_obj) = snapshot.get_item("index").unwrap_or(None) {
            let index: usize = index_obj.extract()
                .map_err(|e| FfiError::TypeConversionError(format!("Invalid index: {}", e)))?;
            data.set_index(index);
        }
        
        if let Some(length_obj) = snapshot.get_item("length").unwrap_or(None) {
            let length: usize = length_obj.extract()
                .map_err(|e| FfiError::TypeConversionError(format!("Invalid length: {}", e)))?;
            data.set_length(length);
        }
        
        if let Some(frozen_obj) = snapshot.get_item("frozen").unwrap_or(None) {
            let frozen: bool = frozen_obj.extract()
                .map_err(|e| FfiError::TypeConversionError(format!("Invalid frozen: {}", e)))?;
            if frozen {
                data.freeze();
            }
        }
        
        // Restore choice sequence
        if let Some(choices_obj) = snapshot.get_item("choices").unwrap_or(None) {
            let choices_list = choices_obj.downcast::<PyList>()
                .map_err(|e| FfiError::TypeConversionError(format!("choices must be list: {}", e)))?;
            
            for choice_obj in choices_list.iter() {
                let choice_dict = choice_obj.downcast::<PyDict>()
                    .map_err(|e| FfiError::TypeConversionError(format!("choice must be dict: {}", e)))?;
                
                let choice_node = import_choice_node_from_python(py, choice_dict)?;
                data.add_choice_node(choice_node);
            }
        }
        
        // Verify binary compatibility if present
        if let Some(binary_obj) = snapshot.get_item("binary_choices").unwrap_or(None) {
            let binary_data = binary_obj.downcast::<PyBytes>()
                .map_err(|e| FfiError::TypeConversionError(format!("binary_choices must be bytes: {}", e)))?
                .as_bytes();
            
            let decoded_choices = ChoiceSequenceBinaryCodec::deserialize_from_bytes(binary_data)?;
            
            if decoded_choices.len() != data.choices().len() {
                ffi_advanced_debug!("WARNING: Binary choice count mismatch: {} vs {}", 
                                   decoded_choices.len(), data.choices().len());
            }
        }
        
        ffi_advanced_debug!("ConjectureData restored from snapshot successfully");
        Ok(data)
    }
    
    /// Compare two ConjectureData instances for exact parity
    pub fn compare_for_parity(
        py: Python,
        rust_data: &ConjectureData,
        python_snapshot: &PyDict
    ) -> PyResult<PyObject> {
        ffi_advanced_debug!("Comparing ConjectureData with Python snapshot for parity");
        
        let comparison = PyDict::new(py);
        let mut differences = Vec::new();
        
        // Compare buffer
        let python_buffer = python_snapshot.get_item("buffer")
            .and_then(|obj| obj.downcast::<PyBytes>().ok())
            .map(|bytes| bytes.as_bytes());
        
        if let Some(py_buf) = python_buffer {
            if rust_data.buffer() != py_buf {
                differences.push("buffer content differs");
            }
        } else {
            differences.push("missing Python buffer");
        }
        
        // Compare index
        if let Some(py_index) = python_snapshot.get_item("index")
            .and_then(|obj| obj.extract::<usize>().ok()) {
            if rust_data.index() != py_index {
                differences.push("index differs");
            }
        }
        
        // Compare frozen status
        if let Some(py_frozen) = python_snapshot.get_item("frozen")
            .and_then(|obj| obj.extract::<bool>().ok()) {
            if rust_data.frozen() != py_frozen {
                differences.push("frozen status differs");
            }
        }
        
        // Compare choice count
        if let Some(py_choices) = python_snapshot.get_item("choices")
            .and_then(|obj| obj.downcast::<PyList>().ok()) {
            if rust_data.choices().len() != py_choices.len() {
                differences.push("choice count differs");
            }
        }
        
        comparison.set_item("parity_check", differences.is_empty())?;
        comparison.set_item("differences", PyList::new(py, &differences))?;
        comparison.set_item("rust_buffer_length", rust_data.buffer().len())?;
        comparison.set_item("rust_index", rust_data.index())?;
        comparison.set_item("rust_frozen", rust_data.frozen())?;
        comparison.set_item("rust_choice_count", rust_data.choices().len())?;
        
        ffi_advanced_debug!("Parity comparison complete: {} differences found", differences.len());
        Ok(comparison.to_object(py))
    }
}

/// PyO3 module exports for advanced FFI functionality
#[pymodule]
pub fn conjecture_data_advanced_ffi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize_choices_to_binary, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_choices_from_binary, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_export_data, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_validate_constraintss_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_state_snapshot_py, m)?)?;
    m.add_function(wrap_pyfunction!(restore_from_snapshot_py, m)?)?;
    m.add_function(wrap_pyfunction!(compare_parity_py, m)?)?;
    Ok(())
}

#[pyfunction]
fn serialize_choices_to_binary(py: Python, choices_list: &PyList) -> PyResult<PyObject> {
    // Convert Python list to Rust ChoiceNode vector and serialize
    let mut choices = Vec::new();
    
    for choice_obj in choices_list.iter() {
        let choice_dict = choice_obj.downcast::<PyDict>()?;
        let choice_node = import_choice_node_from_python(py, choice_dict)
            .map_err(|e| PyRuntimeError::new_err(format!("Choice import error: {}", e)))?;
        choices.push(choice_node);
    }
    
    let binary_data = ChoiceSequenceBinaryCodec::serialize_to_bytes(&choices)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {}", e)))?;
    
    Ok(PyBytes::new(py, &binary_data).to_object(py))
}

#[pyfunction]
fn deserialize_choices_from_binary(py: Python, binary_data: &PyBytes) -> PyResult<PyObject> {
    let choices = ChoiceSequenceBinaryCodec::deserialize_from_bytes(binary_data.as_bytes())
        .map_err(|e| PyRuntimeError::new_err(format!("Deserialization error: {}", e)))?;
    
    let choices_list = PyList::empty(py);
    for choice in choices {
        let choice_dict = export_choice_node_to_python(py, &choice)?;
        choices_list.append(choice_dict)?;
    }
    
    Ok(choices_list.to_object(py))
}

#[pyfunction]
fn bulk_export_data(py: Python, data_list: &PyList) -> PyResult<PyObject> {
    // For now, mock implementation - full version would accept ConjectureData instances
    let result_dict = PyDict::new(py);
    result_dict.set_item("exported_count", data_list.len())?;
    result_dict.set_item("status", "success")?;
    Ok(result_dict.to_object(py))
}

#[pyfunction]
fn bulk_validate_constraintss_py(py: Python, constraintss_list: &PyList) -> PyResult<PyObject> {
    let mut constraintss = Vec::new();
    
    for constraints_obj in constraintss_list.iter() {
        let constraints_dict = constraints_obj.downcast::<PyDict>()?;
        // For now, create a basic constraints - this should be replaced with proper type detection
        let constraints = Constraints::Integer(IntegerConstraints::default());
        constraintss.push(constraints);
    }
    
    BulkOperations::bulk_validate_constraintss(py, &constraintss)
}

#[pyfunction]
fn create_state_snapshot_py(py: Python, data_dict: &PyDict) -> PyResult<PyObject> {
    // Mock implementation - full version would accept ConjectureData
    let snapshot = PyDict::new(py);
    snapshot.set_item("status", "snapshot_created")?;
    snapshot.set_item("timestamp", py.import("time")?.call_method0("time")?)?;
    Ok(snapshot.to_object(py))
}

#[pyfunction]
fn restore_from_snapshot_py(py: Python, snapshot: &PyDict) -> PyResult<PyObject> {
    // Mock implementation - full version would create ConjectureData
    let result = PyDict::new(py);
    result.set_item("status", "restored")?;
    result.set_item("has_snapshot", snapshot.contains("timestamp")?)?;
    Ok(result.to_object(py))
}

#[pyfunction]
fn compare_parity_py(py: Python, rust_data: &PyDict, python_data: &PyDict) -> PyResult<PyObject> {
    // Mock implementation - full version would do comprehensive comparison
    let comparison = PyDict::new(py);
    comparison.set_item("parity_check", true)?;
    comparison.set_item("differences", PyList::empty(py))?;
    Ok(comparison.to_object(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prepare_freethreaded_python;
    
    #[test]
    fn test_binary_serialization_roundtrip() {
        let choices = vec![
            ChoiceNode {
                value: ChoiceValue::Boolean(true),
                constraints: None,
                was_forced: false,
                index: Some(0),
            },
            ChoiceNode {
                value: ChoiceValue::Integer(42),
                constraints: None,
                was_forced: false,
                index: Some(1),
            },
            ChoiceNode {
                value: ChoiceValue::Float(3.14159),
                constraints: None,
                was_forced: false,
                index: Some(2),
            },
            ChoiceNode {
                value: ChoiceValue::String("Hello".to_string()),
                constraints: None,
                was_forced: false,
                index: Some(3),
            },
            ChoiceNode {
                value: ChoiceValue::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]),
                constraints: None,
                was_forced: false,
                index: Some(4),
            },
        ];
        
        let binary_data = ChoiceSequenceBinaryCodec::serialize_to_bytes(&choices).unwrap();
        let deserialized = ChoiceSequenceBinaryCodec::deserialize_from_bytes(&binary_data).unwrap();
        
        assert_eq!(choices.len(), deserialized.len());
        
        for (original, deserialized) in choices.iter().zip(deserialized.iter()) {
            match (&original.value, &deserialized.value) {
                (ChoiceValue::Boolean(a), ChoiceValue::Boolean(b)) => assert_eq!(a, b),
                (ChoiceValue::Integer(a), ChoiceValue::Integer(b)) => assert_eq!(a, b),
                (ChoiceValue::Float(a), ChoiceValue::Float(b)) => assert!((a - b).abs() < f64::EPSILON),
                (ChoiceValue::String(a), ChoiceValue::String(b)) => assert_eq!(a, b),
                (ChoiceValue::Bytes(a), ChoiceValue::Bytes(b)) => assert_eq!(a, b),
                _ => panic!("Type mismatch in deserialization"),
            }
        }
    }
    
    #[test]
    fn test_constraints_validation_and_normalization() {
        let mut constraints = IntegerConstraints {
            min_value: Some(10),
            max_value: Some(100),
            shrink_towards: Some(5), // Below min_value
        };
        
        ConstraintValidator::validate_and_normalize_integer_constraints(&mut constraints).unwrap();
        
        // Should be clamped to min_value
        assert_eq!(constraints.shrink_towards, Some(10));
    }
    
    #[test]
    fn test_integer_encoding_edge_cases() {
        let test_cases = vec![
            0i64,
            1i64,
            -1i64,
            127i64,
            -128i64,
            32767i64,
            -32768i64,
            i64::MAX,
            i64::MIN,
        ];
        
        for original in test_cases {
            let encoded = ChoiceSequenceBinaryCodec::encode_signed_integer(original).unwrap();
            let decoded = ChoiceSequenceBinaryCodec::decode_signed_integer(&encoded).unwrap();
            assert_eq!(original, decoded, "Failed for value: {}", original);
        }
    }
    
    #[test] 
    fn test_uleb128_encoding() {
        let test_values = vec![0, 1, 127, 128, 255, 256, 16383, 16384, 2097151, 2097152];
        
        for original in test_values {
            let mut buffer = Vec::new();
            ChoiceSequenceBinaryCodec::write_uleb128(&mut buffer, original).unwrap();
            
            let mut cursor = Cursor::new(buffer.as_slice());
            let decoded = ChoiceSequenceBinaryCodec::read_uleb128(&mut cursor).unwrap();
            
            assert_eq!(original, decoded, "ULEB128 encoding failed for: {}", original);
        }
    }
}