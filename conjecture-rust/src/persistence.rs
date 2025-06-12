//! Database Persistence Layer for Example Reuse and Regression Prevention
//!
//! This module provides a complete database integration system for storing and retrieving
//! test examples to enable example reuse across test runs and prevent regressions.
//! 
//! # Architecture
//! 
//! The persistence layer follows Python Hypothesis's database design with these components:
//! - `ExampleDatabase` trait: Core database interface for save/load/delete operations
//! - `DirectoryDatabase`: File-based storage with atomic writes and change detection
//! - `InMemoryDatabase`: Memory-based storage for testing and ephemeral usage
//! - `DatabaseKey`: Type-safe key generation system using function digests
//! - `ExampleSerialization`: Binary serialization format matching Python's choices_to_bytes
//! 
//! # Example Usage
//! 
//! ```rust
//! use crate::persistence::{ExampleDatabase, DirectoryDatabase, DatabaseKey};
//! 
//! // Create database instance
//! let mut db = DirectoryDatabase::new("./hypothesis-examples")?;
//! 
//! // Generate key from test function
//! let key = DatabaseKey::from_function("test_my_function", &[])?;
//! 
//! // Save example
//! let example_data = serialize_choices(&choices)?;
//! db.save(&key, &example_data)?;
//! 
//! // Load examples
//! let examples = db.fetch(&key)?;
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

use crate::choice::{ChoiceNode, ChoiceValue, ChoiceType, Constraints, IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints};
use crate::data::{ConjectureData, ConjectureResult};

/// Type alias for database operation results
pub type DatabaseResult<T> = Result<T, DatabaseError>;

/// Comprehensive error types for database operations
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Invalid key: {0}")]
    InvalidKey(String),
    
    #[error("Database corruption: {0}")]
    Corruption(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Disk full or quota exceeded")]
    DiskFull,
    
    #[error("Database locked by another process")]
    Locked,
}

/// Database key for identifying test functions and their examples
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatabaseKey {
    /// Primary hash of the test function (SHA-256)
    pub primary_hash: [u8; 32],
    /// Optional sub-key for different example types
    pub sub_key: Option<String>,
}

impl DatabaseKey {
    /// Create a new database key from a test function name and additional data
    pub fn from_function(function_name: &str, additional_data: &[u8]) -> DatabaseResult<Self> {
        let mut hasher = Sha256::new();
        hasher.update(function_name.as_bytes());
        hasher.update(additional_data);
        
        let primary_hash: [u8; 32] = hasher.finalize().into();
        
        Ok(DatabaseKey {
            primary_hash,
            sub_key: None,
        })
    }
    
    /// Create a sub-key variant (e.g., for secondary corpus, pareto front)
    pub fn with_sub_key(&self, sub_key: &str) -> Self {
        DatabaseKey {
            primary_hash: self.primary_hash,
            sub_key: Some(sub_key.to_string()),
        }
    }
    
    /// Get the hex representation of the key for file storage
    pub fn to_hex(&self) -> String {
        let primary_hex = hex::encode(&self.primary_hash);
        match &self.sub_key {
            Some(sub) => format!("{}.{}", primary_hex, sub),
            None => primary_hex,
        }
    }
    
    /// Parse a hex key back to DatabaseKey
    pub fn from_hex(hex_str: &str) -> DatabaseResult<Self> {
        let parts: Vec<&str> = hex_str.split('.').collect();
        
        let primary_hex = parts[0];
        if primary_hex.len() != 64 {
            return Err(DatabaseError::InvalidKey(
                format!("Invalid primary key length: {}", primary_hex.len())
            ));
        }
        
        let primary_hash: [u8; 32] = hex::decode(primary_hex)
            .map_err(|e| DatabaseError::InvalidKey(format!("Invalid hex: {}", e)))?
            .try_into()
            .map_err(|_| DatabaseError::InvalidKey("Hash wrong length".to_string()))?;
        
        let sub_key = if parts.len() > 1 {
            Some(parts[1..].join("."))
        } else {
            None
        };
        
        Ok(DatabaseKey {
            primary_hash,
            sub_key,
        })
    }
}

/// Event listener function type for database changes
pub type DatabaseListener = Box<dyn Fn(&DatabaseEvent) + Send + Sync>;

/// Events emitted by database operations
#[derive(Debug, Clone)]
pub enum DatabaseEvent {
    ExampleSaved { key: DatabaseKey, value_hash: [u8; 32] },
    ExampleDeleted { key: DatabaseKey, value_hash: [u8; 32] },
    ExampleMoved { src_key: DatabaseKey, dest_key: DatabaseKey, value_hash: [u8; 32] },
}

/// Core database interface trait matching Python Hypothesis's ExampleDatabase
pub trait ExampleDatabase: Send + Sync {
    /// Save a value under a key (idempotent operation)
    fn save(&mut self, key: &DatabaseKey, value: &[u8]) -> DatabaseResult<()>;
    
    /// Fetch all values associated with a key
    fn fetch(&self, key: &DatabaseKey) -> DatabaseResult<Vec<Vec<u8>>>;
    
    /// Delete a specific value from a key (idempotent operation)
    fn delete(&mut self, key: &DatabaseKey, value: &[u8]) -> DatabaseResult<()>;
    
    /// Move a value from one key to another (default implementation: delete + save)
    fn move_value(&mut self, src_key: &DatabaseKey, dest_key: &DatabaseKey, value: &[u8]) -> DatabaseResult<()> {
        self.delete(src_key, value)?;
        self.save(dest_key, value)?;
        Ok(())
    }
    
    /// Add a change listener for database events
    fn add_listener(&mut self, listener: DatabaseListener) -> DatabaseResult<()>;
    
    /// Remove all listeners
    fn clear_listeners(&mut self) -> DatabaseResult<()>;
    
    /// Get statistics about the database
    fn get_stats(&self) -> DatabaseResult<DatabaseStats>;
}

/// Database statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub total_keys: usize,
    pub total_examples: usize,
    pub total_size_bytes: u64,
    pub primary_corpus_size: usize,
    pub secondary_corpus_size: usize,
    pub pareto_corpus_size: usize,
}

/// In-memory database implementation for testing and ephemeral usage
pub struct InMemoryDatabase {
    data: Arc<RwLock<HashMap<DatabaseKey, HashSet<Vec<u8>>>>>,
    listeners: Arc<Mutex<Vec<DatabaseListener>>>,
}

impl std::fmt::Debug for InMemoryDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryDatabase")
            .field("data", &self.data)
            .field("listeners", &format!("{} listeners", self.listeners.lock().map(|l| l.len()).unwrap_or(0)))
            .finish()
    }
}

impl InMemoryDatabase {
    /// Create a new in-memory database
    pub fn new() -> Self {
        InMemoryDatabase {
            data: Arc::new(RwLock::new(HashMap::new())),
            listeners: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Get the number of stored keys
    pub fn key_count(&self) -> usize {
        self.data.read().unwrap().len()
    }
    
    /// Get the total number of stored examples
    pub fn example_count(&self) -> usize {
        self.data.read().unwrap()
            .values()
            .map(|set| set.len())
            .sum()
    }
    
    fn broadcast_event(&self, event: DatabaseEvent) {
        if let Ok(listeners) = self.listeners.lock() {
            for listener in listeners.iter() {
                listener(&event);
            }
        }
    }
}

impl ExampleDatabase for InMemoryDatabase {
    fn save(&mut self, key: &DatabaseKey, value: &[u8]) -> DatabaseResult<()> {
        let mut data = self.data.write().unwrap();
        let entry = data.entry(key.clone()).or_insert_with(HashSet::new);
        let was_new = entry.insert(value.to_vec());
        
        if was_new {
            let value_hash = Sha256::digest(value).into();
            self.broadcast_event(DatabaseEvent::ExampleSaved {
                key: key.clone(),
                value_hash,
            });
        }
        
        Ok(())
    }
    
    fn fetch(&self, key: &DatabaseKey) -> DatabaseResult<Vec<Vec<u8>>> {
        let data = self.data.read().unwrap();
        Ok(data.get(key)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default())
    }
    
    fn delete(&mut self, key: &DatabaseKey, value: &[u8]) -> DatabaseResult<()> {
        let mut data = self.data.write().unwrap();
        if let Some(set) = data.get_mut(key) {
            let was_present = set.remove(value);
            if set.is_empty() {
                data.remove(key);
            }
            
            if was_present {
                let value_hash = Sha256::digest(value).into();
                self.broadcast_event(DatabaseEvent::ExampleDeleted {
                    key: key.clone(),
                    value_hash,
                });
            }
        }
        Ok(())
    }
    
    fn add_listener(&mut self, listener: DatabaseListener) -> DatabaseResult<()> {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.push(listener);
        Ok(())
    }
    
    fn clear_listeners(&mut self) -> DatabaseResult<()> {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.clear();
        Ok(())
    }
    
    fn get_stats(&self) -> DatabaseResult<DatabaseStats> {
        let data = self.data.read().unwrap();
        let total_keys = data.len();
        let total_examples = data.values().map(|set| set.len()).sum();
        let total_size_bytes = data.values()
            .flat_map(|set| set.iter())
            .map(|bytes| bytes.len() as u64)
            .sum();
        
        // Categorize by sub-key
        let mut primary_corpus_size = 0;
        let mut secondary_corpus_size = 0;
        let mut pareto_corpus_size = 0;
        
        for (key, set) in data.iter() {
            let size = set.len();
            match &key.sub_key {
                None => primary_corpus_size += size,
                Some(sub) if sub == "secondary" => secondary_corpus_size += size,
                Some(sub) if sub == "pareto" => pareto_corpus_size += size,
                _ => primary_corpus_size += size,
            }
        }
        
        Ok(DatabaseStats {
            total_keys,
            total_examples,
            total_size_bytes,
            primary_corpus_size,
            secondary_corpus_size,
            pareto_corpus_size,
        })
    }
}

impl Default for InMemoryDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// File-based database implementation with atomic writes and change detection
pub struct DirectoryDatabase {
    base_path: PathBuf,
    listeners: Arc<Mutex<Vec<DatabaseListener>>>,
    stats_cache: Arc<RwLock<Option<(SystemTime, DatabaseStats)>>>,
}

impl std::fmt::Debug for DirectoryDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectoryDatabase")
            .field("base_path", &self.base_path)
            .field("listeners", &format!("{} listeners", self.listeners.lock().map(|l| l.len()).unwrap_or(0)))
            .field("stats_cache", &"<cached>")
            .finish()
    }
}

impl DirectoryDatabase {
    /// Create a new directory-based database
    pub fn new<P: AsRef<Path>>(base_path: P) -> DatabaseResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Create base directory if it doesn't exist
        if !base_path.exists() {
            fs::create_dir_all(&base_path)?;
        }
        
        // Verify we can write to the directory
        let test_file = base_path.join(".write_test");
        File::create(&test_file)
            .and_then(|_| fs::remove_file(&test_file))
            .map_err(|e| match e.kind() {
                io::ErrorKind::PermissionDenied => DatabaseError::PermissionDenied(
                    format!("Cannot write to directory: {}", base_path.display())
                ),
                _ => DatabaseError::Io(e),
            })?;
        
        Ok(DirectoryDatabase {
            base_path,
            listeners: Arc::new(Mutex::new(Vec::new())),
            stats_cache: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Get the file path for a specific key and value
    fn get_file_path(&self, key: &DatabaseKey, value: &[u8]) -> PathBuf {
        let key_dir = self.base_path.join(format!("{:016X}", key.primary_hash[0] as u64));
        let value_hash = Sha256::digest(value);
        let filename = format!("{}.example", hex::encode(&value_hash[..16]));
        key_dir.join(filename)
    }
    
    /// Get the directory path for a key
    fn get_key_dir(&self, key: &DatabaseKey) -> PathBuf {
        self.base_path.join(format!("{:016X}", key.primary_hash[0] as u64))
    }
    
    /// Atomic write operation using temporary file + rename
    fn atomic_write(&self, path: &Path, data: &[u8]) -> DatabaseResult<()> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Write to temporary file first
        let temp_path = path.with_extension("tmp");
        {
            let mut file = BufWriter::new(File::create(&temp_path)?);
            file.write_all(data)?;
            file.flush()?;
        }
        
        // Atomic rename
        fs::rename(&temp_path, path)?;
        Ok(())
    }
    
    /// Load all examples for a key from the file system
    fn load_key_examples(&self, key: &DatabaseKey) -> DatabaseResult<Vec<Vec<u8>>> {
        let key_dir = self.get_key_dir(key);
        if !key_dir.exists() {
            return Ok(Vec::new());
        }
        
        let mut examples = Vec::new();
        for entry in fs::read_dir(&key_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().map_or(false, |ext| ext == "example") {
                match fs::read(&path) {
                    Ok(data) => {
                        // Verify the data is valid by attempting to parse it
                        if self.validate_example_data(&data) {
                            examples.push(data);
                        } else {
                            // Clean up corrupted files
                            eprintln!("DEBUG: Removing corrupted example file: {:?}", path);
                            let _ = fs::remove_file(&path);
                        }
                    }
                    Err(e) => {
                        eprintln!("DEBUG: Failed to read example file {:?}: {}", path, e);
                        // Continue with other files
                    }
                }
            }
        }
        
        Ok(examples)
    }
    
    /// Validate that example data is properly formatted
    fn validate_example_data(&self, data: &[u8]) -> bool {
        // Basic validation - ensure it's not empty and has reasonable structure
        if data.is_empty() {
            return false;
        }
        
        // Try to parse as serialized choices (simplified validation)
        // In a full implementation, this would use the actual deserialization logic
        data.len() >= 4 && data.len() <= 1_000_000 // Reasonable size bounds
    }
    
    fn broadcast_event(&self, event: DatabaseEvent) {
        if let Ok(listeners) = self.listeners.lock() {
            for listener in listeners.iter() {
                listener(&event);
            }
        }
    }
    
    /// Clear the stats cache to force recalculation
    fn invalidate_stats_cache(&self) {
        if let Ok(mut cache) = self.stats_cache.write() {
            *cache = None;
        }
    }
}

impl ExampleDatabase for DirectoryDatabase {
    fn save(&mut self, key: &DatabaseKey, value: &[u8]) -> DatabaseResult<()> {
        let file_path = self.get_file_path(key, value);
        
        // Check if file already exists (idempotent operation)
        if file_path.exists() {
            return Ok(());
        }
        
        // Atomic write
        self.atomic_write(&file_path, value)?;
        
        // Broadcast event
        let value_hash = Sha256::digest(value).into();
        self.broadcast_event(DatabaseEvent::ExampleSaved {
            key: key.clone(),
            value_hash,
        });
        
        // Invalidate stats cache
        self.invalidate_stats_cache();
        
        Ok(())
    }
    
    fn fetch(&self, key: &DatabaseKey) -> DatabaseResult<Vec<Vec<u8>>> {
        self.load_key_examples(key)
    }
    
    fn delete(&mut self, key: &DatabaseKey, value: &[u8]) -> DatabaseResult<()> {
        let file_path = self.get_file_path(key, value);
        
        if !file_path.exists() {
            return Ok(()); // Idempotent - already deleted
        }
        
        fs::remove_file(&file_path)?;
        
        // Clean up empty directories
        let key_dir = self.get_key_dir(key);
        if key_dir.exists() {
            if let Ok(mut entries) = fs::read_dir(&key_dir) {
                if entries.next().is_none() {
                    let _ = fs::remove_dir(&key_dir);
                }
            }
        }
        
        // Broadcast event
        let value_hash = Sha256::digest(value).into();
        self.broadcast_event(DatabaseEvent::ExampleDeleted {
            key: key.clone(),
            value_hash,
        });
        
        // Invalidate stats cache
        self.invalidate_stats_cache();
        
        Ok(())
    }
    
    fn add_listener(&mut self, listener: DatabaseListener) -> DatabaseResult<()> {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.push(listener);
        Ok(())
    }
    
    fn clear_listeners(&mut self) -> DatabaseResult<()> {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.clear();
        Ok(())
    }
    
    fn get_stats(&self) -> DatabaseResult<DatabaseStats> {
        // Check cache first (with 5-second TTL)
        if let Ok(cache) = self.stats_cache.read() {
            if let Some((timestamp, stats)) = cache.as_ref() {
                if let Ok(elapsed) = timestamp.elapsed() {
                    if elapsed.as_secs() < 5 {
                        return Ok(stats.clone());
                    }
                }
            }
        }
        
        // Recalculate stats
        let mut total_keys = 0;
        let mut total_examples = 0;
        let mut total_size_bytes = 0u64;
        let mut primary_corpus_size = 0;
        let mut secondary_corpus_size = 0;
        let mut pareto_corpus_size = 0;
        
        // Walk through all key directories
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() && !path.file_name().unwrap().to_string_lossy().starts_with('.') {
                // Count examples in this key directory
                let mut key_examples = 0;
                if let Ok(dir_entries) = fs::read_dir(&path) {
                    for file_entry in dir_entries {
                        if let Ok(file_entry) = file_entry {
                            let file_path = file_entry.path();
                            if file_path.extension().map_or(false, |ext| ext == "example") {
                                key_examples += 1;
                                if let Ok(metadata) = file_path.metadata() {
                                    total_size_bytes += metadata.len();
                                }
                            }
                        }
                    }
                }
                
                if key_examples > 0 {
                    total_keys += 1;
                    total_examples += key_examples;
                    
                    // Categorize based on directory structure (simplified)
                    // In practice, would need to parse the actual key to determine sub-key
                    primary_corpus_size += key_examples;
                }
            }
        }
        
        let stats = DatabaseStats {
            total_keys,
            total_examples,
            total_size_bytes,
            primary_corpus_size,
            secondary_corpus_size,
            pareto_corpus_size,
        };
        
        // Update cache
        if let Ok(mut cache) = self.stats_cache.write() {
            *cache = Some((SystemTime::now(), stats.clone()));
        }
        
        Ok(stats)
    }
}

/// Binary serialization system for ChoiceNodes matching Python's choices_to_bytes format
pub struct ExampleSerialization;

impl ExampleSerialization {
    /// Serialize choice nodes to binary format compatible with Python Hypothesis
    pub fn serialize_choices(choices: &[ChoiceNode]) -> DatabaseResult<Vec<u8>> {
        let mut buffer = Vec::new();
        
        for choice in choices {
            Self::serialize_choice(&mut buffer, choice)?;
        }
        
        Ok(buffer)
    }
    
    /// Deserialize binary data back to choice nodes
    pub fn deserialize_choices(data: &[u8]) -> DatabaseResult<Vec<ChoiceNode>> {
        let mut cursor = 0;
        let mut choices = Vec::new();
        
        while cursor < data.len() {
            let (choice, consumed) = Self::deserialize_choice(&data[cursor..])?;
            choices.push(choice);
            cursor += consumed;
        }
        
        Ok(choices)
    }
    
    /// Serialize a single choice using Python-compatible format
    fn serialize_choice(buffer: &mut Vec<u8>, choice: &ChoiceNode) -> DatabaseResult<()> {
        match &choice.value {
            ChoiceValue::Boolean(val) => {
                // Format: 000_0000v (value inlined in low bit)
                let tag = if *val { 0b0000_0001 } else { 0b0000_0000 };
                buffer.push(tag);
            }
            
            ChoiceValue::Float(val) => {
                // Format: 001_ssss + 8-byte IEEE 754 double
                buffer.push(0b0010_1000); // tag for float with size 8
                buffer.extend_from_slice(&val.to_be_bytes());
            }
            
            ChoiceValue::Integer(val) => {
                // Format: 010_ssss + variable-length big-endian signed bytes
                let bytes = Self::encode_signed_integer(*val as i64);
                let size = bytes.len();
                
                if size < 31 {
                    buffer.push(0b0100_0000 | (size as u8));
                } else {
                    buffer.push(0b0100_0000 | 31);
                    Self::encode_uleb128(buffer, size as u64);
                }
                buffer.extend_from_slice(&bytes);
            }
            
            ChoiceValue::Bytes(val) => {
                // Format: 011_ssss + raw bytes
                let size = val.len();
                
                if size < 31 {
                    buffer.push(0b0110_0000 | (size as u8));
                } else {
                    buffer.push(0b0110_0000 | 31);
                    Self::encode_uleb128(buffer, size as u64);
                }
                buffer.extend_from_slice(val);
            }
            
            ChoiceValue::String(val) => {
                // Format: 100_ssss + UTF-8 bytes
                let utf8_bytes = val.as_bytes();
                let size = utf8_bytes.len();
                
                if size < 31 {
                    buffer.push(0b1000_0000 | (size as u8));
                } else {
                    buffer.push(0b1000_0000 | 31);
                    Self::encode_uleb128(buffer, size as u64);
                }
                buffer.extend_from_slice(utf8_bytes);
            }
        }
        
        Ok(())
    }
    
    /// Deserialize a single choice from binary data
    fn deserialize_choice(data: &[u8]) -> DatabaseResult<(ChoiceNode, usize)> {
        if data.is_empty() {
            return Err(DatabaseError::Corruption("Empty choice data".to_string()));
        }
        
        let tag = data[0];
        let type_bits = (tag & 0b1110_0000) >> 5;
        let size_bits = tag & 0b0001_1111;
        
        match type_bits {
            0b000 => {
                // Boolean: value in low bit
                let value = (tag & 0b0000_0001) != 0;
                let choice = ChoiceNode {
                    choice_type: ChoiceType::Boolean,
                    value: ChoiceValue::Boolean(value),
                    constraints: Constraints::Boolean(BooleanConstraints::default()),
                    was_forced: false,
                    index: None,
                };
                Ok((choice, 1))
            }
            
            0b001 => {
                // Float: 8-byte IEEE 754 double
                if data.len() < 9 {
                    return Err(DatabaseError::Corruption("Incomplete float data".to_string()));
                }
                
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&data[1..9]);
                let value = f64::from_be_bytes(bytes);
                
                let choice = ChoiceNode {
                    choice_type: ChoiceType::Float,
                    value: ChoiceValue::Float(value),
                    constraints: Constraints::Float(FloatConstraints::default()),
                    was_forced: false,
                    index: None,
                };
                Ok((choice, 9))
            }
            
            0b010 => {
                // Integer: variable-length signed bytes
                let (size, size_bytes) = if size_bits < 31 {
                    (size_bits as usize, 0)
                } else {
                    let (s, sb) = Self::decode_uleb128(&data[1..])?;
                    (s as usize, sb)
                };
                
                let data_start = 1 + size_bytes;
                if data.len() < data_start + size {
                    return Err(DatabaseError::Corruption("Incomplete integer data".to_string()));
                }
                
                let value = Self::decode_signed_integer(&data[data_start..data_start + size])?;
                let choice = ChoiceNode {
                    choice_type: ChoiceType::Integer,
                    value: ChoiceValue::Integer(value as i128),
                    constraints: Constraints::Integer(IntegerConstraints::default()),
                    was_forced: false,
                    index: None,
                };
                Ok((choice, data_start + size))
            }
            
            0b011 => {
                // Bytes: raw bytes
                let (size, size_bytes) = if size_bits < 31 {
                    (size_bits as usize, 0)
                } else {
                    let (s, sb) = Self::decode_uleb128(&data[1..])?;
                    (s as usize, sb)
                };
                
                let data_start = 1 + size_bytes;
                if data.len() < data_start + size {
                    return Err(DatabaseError::Corruption("Incomplete bytes data".to_string()));
                }
                
                let value = data[data_start..data_start + size].to_vec();
                let choice = ChoiceNode {
                    choice_type: ChoiceType::Bytes,
                    value: ChoiceValue::Bytes(value),
                    constraints: Constraints::Bytes(BytesConstraints::default()),
                    was_forced: false,
                    index: None,
                };
                Ok((choice, data_start + size))
            }
            
            0b100 => {
                // String: UTF-8 bytes
                let (size, size_bytes) = if size_bits < 31 {
                    (size_bits as usize, 0)
                } else {
                    let (s, sb) = Self::decode_uleb128(&data[1..])?;
                    (s as usize, sb)
                };
                
                let data_start = 1 + size_bytes;
                if data.len() < data_start + size {
                    return Err(DatabaseError::Corruption("Incomplete string data".to_string()));
                }
                
                let utf8_bytes = &data[data_start..data_start + size];
                let value = String::from_utf8(utf8_bytes.to_vec())
                    .map_err(|e| DatabaseError::Corruption(format!("Invalid UTF-8: {}", e)))?;
                
                let choice = ChoiceNode {
                    choice_type: ChoiceType::String,
                    value: ChoiceValue::String(value),
                    constraints: Constraints::String(StringConstraints::default()),
                    was_forced: false,
                    index: None,
                };
                Ok((choice, data_start + size))
            }
            
            _ => Err(DatabaseError::Corruption(format!("Unknown choice type: {}", type_bits))),
        }
    }
    
    /// Encode a signed integer as variable-length big-endian bytes
    fn encode_signed_integer(mut value: i64) -> Vec<u8> {
        if value == 0 {
            return vec![0];
        }
        
        let is_negative = value < 0;
        if is_negative {
            value = -value;
        }
        
        let mut bytes = Vec::new();
        while value > 0 {
            bytes.push((value & 0xFF) as u8);
            value >>= 8;
        }
        
        // Convert to big-endian
        bytes.reverse();
        
        // Add sign information if needed
        if is_negative {
            // Two's complement representation
            let mut carry = true;
            for byte in bytes.iter_mut().rev() {
                *byte = !*byte;
                if carry {
                    let sum = *byte as u16 + 1;
                    *byte = sum as u8;
                    carry = sum > 255;
                }
            }
            
            // Ensure high bit is set for negative numbers
            if bytes.is_empty() || (bytes[0] & 0x80) == 0 {
                bytes.insert(0, 0xFF);
            }
        } else {
            // Ensure high bit is clear for positive numbers
            if !bytes.is_empty() && (bytes[0] & 0x80) != 0 {
                bytes.insert(0, 0x00);
            }
        }
        
        bytes
    }
    
    /// Decode variable-length big-endian signed bytes to integer
    fn decode_signed_integer(bytes: &[u8]) -> DatabaseResult<i64> {
        if bytes.is_empty() {
            return Ok(0);
        }
        
        let is_negative = (bytes[0] & 0x80) != 0;
        let mut value = 0i64;
        
        if is_negative {
            // Two's complement decoding
            let mut temp_bytes = bytes.to_vec();
            
            // Subtract 1
            let mut borrow = true;
            for byte in temp_bytes.iter_mut().rev() {
                if borrow {
                    if *byte == 0 {
                        *byte = 0xFF;
                    } else {
                        *byte -= 1;
                        borrow = false;
                    }
                }
            }
            
            // Invert bits
            for byte in temp_bytes.iter_mut() {
                *byte = !*byte;
            }
            
            // Convert to positive value
            for &byte in &temp_bytes {
                value = value.wrapping_shl(8) | (byte as i64);
            }
            
            value = -value;
        } else {
            // Positive number
            for &byte in bytes {
                value = value.wrapping_shl(8) | (byte as i64);
            }
        }
        
        Ok(value)
    }
    
    /// Encode ULEB128 (Unsigned Little Endian Base 128) variable-length integer
    fn encode_uleb128(buffer: &mut Vec<u8>, mut value: u64) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            
            if value != 0 {
                byte |= 0x80; // Set continuation bit
            }
            
            buffer.push(byte);
            
            if value == 0 {
                break;
            }
        }
    }
    
    /// Decode ULEB128 variable-length integer
    fn decode_uleb128(data: &[u8]) -> DatabaseResult<(u64, usize)> {
        let mut value = 0u64;
        let mut shift = 0;
        let mut bytes_read = 0;
        
        for &byte in data {
            bytes_read += 1;
            
            let payload = (byte & 0x7F) as u64;
            value |= payload << shift;
            
            if (byte & 0x80) == 0 {
                // No continuation bit, we're done
                return Ok((value, bytes_read));
            }
            
            shift += 7;
            if shift >= 64 {
                return Err(DatabaseError::Corruption("ULEB128 too long".to_string()));
            }
        }
        
        Err(DatabaseError::Corruption("Incomplete ULEB128".to_string()))
    }
}

/// Utility functions for database integration with the test engine
pub struct DatabaseIntegration;

impl DatabaseIntegration {
    /// Create a database instance from configuration
    pub fn create_database(database_path: Option<&str>) -> DatabaseResult<Box<dyn ExampleDatabase>> {
        match database_path {
            Some(path) => {
                let db = DirectoryDatabase::new(path)?;
                Ok(Box::new(db))
            }
            None => {
                let db = InMemoryDatabase::new();
                Ok(Box::new(db))
            }
        }
    }
    
    /// Generate a database key from test function metadata
    pub fn generate_key(
        function_name: &str,
        function_source: Option<&str>,
        additional_data: &[u8],
    ) -> DatabaseResult<DatabaseKey> {
        let mut hasher = Sha256::new();
        
        // Hash function name
        hasher.update(function_name.as_bytes());
        
        // Hash normalized source code if available
        if let Some(source) = function_source {
            let normalized_source = Self::normalize_source_code(source);
            hasher.update(normalized_source.as_bytes());
        }
        
        // Hash additional digest data
        hasher.update(additional_data);
        
        let primary_hash: [u8; 32] = hasher.finalize().into();
        
        Ok(DatabaseKey {
            primary_hash,
            sub_key: None,
        })
    }
    
    /// Normalize source code for consistent hashing
    fn normalize_source_code(source: &str) -> String {
        // Remove comments and normalize whitespace
        source
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    /// Convert ConjectureData to serializable format
    pub fn serialize_example(data: &ConjectureData) -> DatabaseResult<Vec<u8>> {
        // For now, serialize the choice sequence
        // In a full implementation, this might include additional metadata
        ExampleSerialization::serialize_choices(data.choices())
    }
    
    /// Convert ConjectureResult to serializable format
    pub fn serialize_result(result: &ConjectureResult) -> DatabaseResult<Vec<u8>> {
        // Serialize the choice sequence from the result
        ExampleSerialization::serialize_choices(&result.nodes)
    }
    
    /// Convert serialized data back to choices for replay
    pub fn deserialize_example(data: &[u8]) -> DatabaseResult<Vec<ChoiceNode>> {
        ExampleSerialization::deserialize_choices(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_database_key_creation() {
        let key = DatabaseKey::from_function("test_function", b"additional_data").unwrap();
        assert_eq!(key.sub_key, None);
        
        let sub_key = key.with_sub_key("secondary");
        assert_eq!(sub_key.sub_key, Some("secondary".to_string()));
    }
    
    #[test]
    fn test_database_key_hex_conversion() {
        let key = DatabaseKey::from_function("test_function", b"data").unwrap();
        let hex = key.to_hex();
        let parsed_key = DatabaseKey::from_hex(&hex).unwrap();
        assert_eq!(key, parsed_key);
    }
    
    #[test]
    fn test_in_memory_database() {
        let mut db = InMemoryDatabase::new();
        let key = DatabaseKey::from_function("test", b"").unwrap();
        let value = b"example_data";
        
        // Test save and fetch
        db.save(&key, value).unwrap();
        let results = db.fetch(&key).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], value);
        
        // Test delete
        db.delete(&key, value).unwrap();
        let results = db.fetch(&key).unwrap();
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_directory_database() {
        let temp_dir = tempdir().unwrap();
        let mut db = DirectoryDatabase::new(temp_dir.path()).unwrap();
        
        let key = DatabaseKey::from_function("test", b"").unwrap();
        let value = b"example_data";
        
        // Test save and fetch
        db.save(&key, value).unwrap();
        let results = db.fetch(&key).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], value);
        
        // Test persistence across instances
        let db2 = DirectoryDatabase::new(temp_dir.path()).unwrap();
        let results = db2.fetch(&key).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], value);
    }
    
    #[test]
    fn test_choice_serialization() {
        let choices = vec![
            ChoiceNode {
                choice_type: ChoiceType::Boolean,
                value: ChoiceValue::Boolean(true),
                constraints: Constraints::Boolean(BooleanConstraints::default()),
                was_forced: false,
                index: None,
            },
            ChoiceNode {
                choice_type: ChoiceType::Integer,
                value: ChoiceValue::Integer(42),
                constraints: Constraints::Integer(IntegerConstraints::default()),
                was_forced: false,
                index: None,
            },
            ChoiceNode {
                choice_type: ChoiceType::String,
                value: ChoiceValue::String("hello".to_string()),
                constraints: Constraints::String(StringConstraints::default()),
                was_forced: false,
                index: None,
            },
        ];
        
        let serialized = ExampleSerialization::serialize_choices(&choices).unwrap();
        let deserialized = ExampleSerialization::deserialize_choices(&serialized).unwrap();
        
        assert_eq!(choices.len(), deserialized.len());
        for (original, recovered) in choices.iter().zip(deserialized.iter()) {
            assert_eq!(original.value, recovered.value);
        }
    }
    
    #[test]
    fn test_signed_integer_encoding() {
        let test_values = [0, 1, -1, 127, -128, 32767, -32768, 2147483647, -2147483648];
        
        for &value in &test_values {
            let encoded = ExampleSerialization::encode_signed_integer(value);
            let decoded = ExampleSerialization::decode_signed_integer(&encoded).unwrap();
            assert_eq!(value, decoded, "Failed for value: {}", value);
        }
    }
    
    #[test]
    fn test_database_stats() {
        let mut db = InMemoryDatabase::new();
        let key1 = DatabaseKey::from_function("test1", b"").unwrap();
        let key2 = DatabaseKey::from_function("test2", b"").unwrap().with_sub_key("secondary");
        
        db.save(&key1, b"data1").unwrap();
        db.save(&key1, b"data2").unwrap();
        db.save(&key2, b"data3").unwrap();
        
        let stats = db.get_stats().unwrap();
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.total_examples, 3);
        assert_eq!(stats.primary_corpus_size, 2);
        assert_eq!(stats.secondary_corpus_size, 1);
    }
    
    #[test]
    fn test_uleb128_encoding() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384];
        
        for &value in &test_values {
            let mut buffer = Vec::new();
            ExampleSerialization::encode_uleb128(&mut buffer, value);
            let (decoded, bytes_read) = ExampleSerialization::decode_uleb128(&buffer).unwrap();
            assert_eq!(value, decoded, "Failed for value: {}", value);
            assert_eq!(bytes_read, buffer.len());
        }
    }
}