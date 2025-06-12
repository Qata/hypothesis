use crate::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, StringConstraints, BytesConstraints};
use crate::providers::{PrimitiveProvider, HypothesisProvider, RandomProvider, GlobalConstants, DrawError, ProviderLifetime};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyString};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, PartialEq)]
pub struct LocalConstant {
    pub value: ChoiceValue,
    pub source_module: String,
    pub discovery_count: usize,
    pub last_used: Instant,
    pub selection_weight: f64,
}

#[derive(Debug, Clone)]
pub struct ConstantDiscoveryStats {
    pub modules_scanned: usize,
    pub constants_discovered: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub injection_attempts: usize,
    pub successful_injections: usize,
}

pub struct LRUConstantCache {
    capacity: usize,
    cache: HashMap<String, LocalConstant>,
    access_order: VecDeque<String>,
    stats: ConstantDiscoveryStats,
}

impl LRUConstantCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            stats: ConstantDiscoveryStats {
                modules_scanned: 0,
                constants_discovered: 0,
                cache_hits: 0,
                cache_misses: 0,
                injection_attempts: 0,
                successful_injections: 0,
            },
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&LocalConstant> {
        if let Some(constant) = self.cache.get(key) {
            self.stats.cache_hits += 1;
            self.move_to_front(key);
            Some(constant)
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: String, constant: LocalConstant) {
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            if let Some(lru_key) = self.access_order.pop_back() {
                self.cache.remove(&lru_key);
            }
        }

        self.cache.insert(key.clone(), constant);
        self.move_to_front(&key);
        self.stats.constants_discovered += 1;
    }

    fn move_to_front(&mut self, key: &str) {
        self.access_order.retain(|k| k != key);
        self.access_order.push_front(key.to_string());
    }

    pub fn stats(&self) -> &ConstantDiscoveryStats {
        &self.stats
    }
}

pub struct AdvancedConstantInjectionProvider {
    base_provider: HypothesisProvider,
    local_constant_cache: Arc<Mutex<LRUConstantCache>>,
    global_constants: GlobalConstants,
    injection_probability_calculator: Box<dyn Fn(&ChoiceType, &Constraints) -> f64 + Send + Sync>,
    constant_selection_strategy: Box<dyn Fn(&[LocalConstant], &Constraints) -> Option<ChoiceValue> + Send + Sync>,
    discovery_enabled: bool,
    performance_metrics: Arc<RwLock<HashMap<String, Duration>>>,
}

impl AdvancedConstantInjectionProvider {
    pub fn new() -> Self {
        Self {
            base_provider: HypothesisProvider::new(),
            local_constant_cache: Arc::new(Mutex::new(LRUConstantCache::new(1000))),
            global_constants: GlobalConstants::new(),
            injection_probability_calculator: Box::new(|choice_type, constraints| {
                match choice_type {
                    ChoiceType::Integer => 0.15,
                    ChoiceType::Float => 0.20,
                    ChoiceType::String => 0.10,
                    ChoiceType::Bytes => 0.08,
                    ChoiceType::Boolean => 0.05,
                }
            }),
            constant_selection_strategy: Box::new(|constants, _constraints| {
                constants.first().map(|c| c.value.clone())
            }),
            discovery_enabled: true,
            performance_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_advanced_selection_strategy(mut self) -> Self {
        self.constant_selection_strategy = Box::new(|constants, constraints| {
            let mut weighted_candidates: Vec<(f64, &LocalConstant)> = constants.iter()
                .map(|c| {
                    let recency_weight = 1.0 / (c.last_used.elapsed().as_secs() as f64 + 1.0);
                    let usage_weight = (c.discovery_count as f64).ln().max(1.0);
                    let base_weight = c.selection_weight;
                    (recency_weight * usage_weight * base_weight, c)
                })
                .collect();

            weighted_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            weighted_candidates.first().map(|(_, constant)| constant.value.clone())
        });
        self
    }

    pub fn discover_local_constants(&mut self, py_module: &PyModule) -> PyResult<usize> {
        let start_time = Instant::now();
        let mut discovered_count = 0;

        let module_dict = py_module.dict();
        let mut cache = self.local_constant_cache.lock().unwrap();
        cache.stats.modules_scanned += 1;

        for (key, value) in module_dict.iter() {
            if let Ok(key_str) = key.extract::<String>() {
                if let Some(constant_value) = self.extract_constant_value(value) {
                    let local_constant = LocalConstant {
                        value: constant_value,
                        source_module: py_module.name()?.to_string(),
                        discovery_count: 1,
                        last_used: Instant::now(),
                        selection_weight: 1.0,
                    };
                    
                    cache.insert(key_str, local_constant);
                    discovered_count += 1;
                }
            }
        }

        let discovery_time = start_time.elapsed();
        self.performance_metrics.write().unwrap()
            .insert("constant_discovery".to_string(), discovery_time);

        Ok(discovered_count)
    }

    fn extract_constant_value(&self, py_value: &PyAny) -> Option<ChoiceValue> {
        if let Ok(int_val) = py_value.extract::<i128>() {
            Some(ChoiceValue::Integer(int_val))
        } else if let Ok(float_val) = py_value.extract::<f64>() {
            Some(ChoiceValue::Float(float_val))
        } else if let Ok(str_val) = py_value.extract::<String>() {
            Some(ChoiceValue::String(str_val))
        } else if let Ok(bool_val) = py_value.extract::<bool>() {
            Some(ChoiceValue::Boolean(bool_val))
        } else if let Ok(bytes_val) = py_value.extract::<Vec<u8>>() {
            Some(ChoiceValue::Bytes(bytes_val))
        } else {
            None
        }
    }

    pub fn maybe_draw_advanced_constant(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Option<ChoiceValue> {
        let start_time = Instant::now();
        
        let injection_probability = (self.injection_probability_calculator)(&choice_type, constraints);
        let should_inject = self.base_provider.rng.gen_bool(injection_probability);

        if !should_inject {
            return None;
        }

        let mut cache = self.local_constant_cache.lock().unwrap();
        cache.stats.injection_attempts += 1;

        let local_candidates: Vec<LocalConstant> = cache.cache.values()
            .filter(|constant| self.constant_matches_type_and_constraints(&constant.value, &choice_type, constraints))
            .cloned()
            .collect();

        let selected_constant = if !local_candidates.is_empty() {
            (self.constant_selection_strategy)(&local_candidates, constraints)
        } else {
            self.select_global_constant(&choice_type, constraints)
        };

        if selected_constant.is_some() {
            cache.stats.successful_injections += 1;
        }

        let selection_time = start_time.elapsed();
        self.performance_metrics.write().unwrap()
            .insert("constant_selection".to_string(), selection_time);

        selected_constant
    }

    fn constant_matches_type_and_constraints(&self, value: &ChoiceValue, choice_type: &ChoiceType, constraints: &Constraints) -> bool {
        match (value, choice_type, constraints) {
            (ChoiceValue::Integer(val), ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                *val >= int_constraints.min && *val <= int_constraints.max
            },
            (ChoiceValue::Float(val), ChoiceType::Float, Constraints::Float(float_constraints)) => {
                if val.is_nan() && !float_constraints.allow_nan {
                    return false;
                }
                *val >= float_constraints.min && *val <= float_constraints.max
            },
            (ChoiceValue::String(val), ChoiceType::String, Constraints::String(str_constraints)) => {
                val.len() >= str_constraints.min_size && val.len() <= str_constraints.max_size
            },
            (ChoiceValue::Bytes(val), ChoiceType::Bytes, Constraints::Bytes(bytes_constraints)) => {
                val.len() >= bytes_constraints.min_size && val.len() <= bytes_constraints.max_size
            },
            (ChoiceValue::Boolean(_), ChoiceType::Boolean, Constraints::Boolean(_)) => true,
            _ => false,
        }
    }

    fn select_global_constant(&self, choice_type: &ChoiceType, constraints: &Constraints) -> Option<ChoiceValue> {
        match choice_type {
            ChoiceType::Integer => {
                if let Constraints::Integer(int_constraints) = constraints {
                    self.global_constants.integers.iter()
                        .find(|&&val| val >= int_constraints.min && val <= int_constraints.max)
                        .map(|&val| ChoiceValue::Integer(val))
                } else {
                    None
                }
            },
            ChoiceType::Float => {
                if let Constraints::Float(float_constraints) = constraints {
                    self.global_constants.floats.iter()
                        .find(|&&val| {
                            if val.is_nan() && !float_constraints.allow_nan {
                                return false;
                            }
                            val >= float_constraints.min && val <= float_constraints.max
                        })
                        .map(|&val| ChoiceValue::Float(val))
                } else {
                    None
                }
            },
            ChoiceType::String => {
                if let Constraints::String(str_constraints) = constraints {
                    self.global_constants.strings.iter()
                        .find(|s| s.len() >= str_constraints.min_size && s.len() <= str_constraints.max_size)
                        .map(|s| ChoiceValue::String(s.clone()))
                } else {
                    None
                }
            },
            ChoiceType::Bytes => {
                if let Constraints::Bytes(bytes_constraints) = constraints {
                    self.global_constants.bytes.iter()
                        .find(|b| b.len() >= bytes_constraints.min_size && b.len() <= bytes_constraints.max_size)
                        .map(|b| ChoiceValue::Bytes(b.clone()))
                } else {
                    None
                }
            },
            ChoiceType::Boolean => Some(ChoiceValue::Boolean(self.base_provider.rng.gen_bool(0.5))),
        }
    }

    pub fn get_discovery_stats(&self) -> ConstantDiscoveryStats {
        self.local_constant_cache.lock().unwrap().stats().clone()
    }

    pub fn get_performance_metrics(&self) -> HashMap<String, Duration> {
        self.performance_metrics.read().unwrap().clone()
    }

    pub fn cache_size(&self) -> usize {
        self.local_constant_cache.lock().unwrap().cache.len()
    }
}

impl PrimitiveProvider for AdvancedConstantInjectionProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestLifetime
    }

    fn draw_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, DrawError> {
        if let Some(constant) = self.maybe_draw_advanced_constant(choice_type.clone(), constraints) {
            Ok(constant)
        } else {
            self.base_provider.draw_choice(choice_type, constraints)
        }
    }

    fn generate_integer(&mut self, min: i128, max: i128) -> Result<i128, DrawError> {
        let constraints = Constraints::Integer(IntegerConstraints::new(min, max));
        if let Some(ChoiceValue::Integer(val)) = self.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints) {
            Ok(val)
        } else {
            self.base_provider.generate_integer(min, max)
        }
    }

    fn generate_boolean(&mut self, probability: Option<f64>) -> Result<bool, DrawError> {
        let constraints = Constraints::Boolean(crate::choice::BooleanConstraints { probability });
        if let Some(ChoiceValue::Boolean(val)) = self.maybe_draw_advanced_constant(ChoiceType::Boolean, &constraints) {
            Ok(val)
        } else {
            self.base_provider.generate_boolean(probability)
        }
    }

    fn generate_float(&mut self, min: f64, max: f64, allow_nan: bool, smallest_nonzero_magnitude: f64) -> Result<f64, DrawError> {
        let constraints = Constraints::Float(FloatConstraints { min, max, allow_nan, smallest_nonzero_magnitude });
        if let Some(ChoiceValue::Float(val)) = self.maybe_draw_advanced_constant(ChoiceType::Float, &constraints) {
            Ok(val)
        } else {
            self.base_provider.generate_float(min, max, allow_nan, smallest_nonzero_magnitude)
        }
    }

    fn generate_string(&mut self, min_size: usize, max_size: usize, intervals: Vec<(u32, u32)>) -> Result<String, DrawError> {
        let constraints = Constraints::String(StringConstraints { min_size, max_size, intervals });
        if let Some(ChoiceValue::String(val)) = self.maybe_draw_advanced_constant(ChoiceType::String, &constraints) {
            Ok(val)
        } else {
            self.base_provider.generate_string(min_size, max_size, intervals)
        }
    }

    fn generate_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, DrawError> {
        let constraints = Constraints::Bytes(BytesConstraints { min_size, max_size });
        if let Some(ChoiceValue::Bytes(val)) = self.maybe_draw_advanced_constant(ChoiceType::Bytes, &constraints) {
            Ok(val)
        } else {
            self.base_provider.generate_bytes(min_size, max_size)
        }
    }

    fn span_start(&mut self, label: String) -> Result<(), DrawError> {
        self.base_provider.span_start(label)
    }

    fn span_end(&mut self, successful: bool) -> Result<(), DrawError> {
        self.base_provider.span_end(successful)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_lru_cache_basic_operations() {
        let mut cache = LRUConstantCache::new(2);
        
        let constant1 = LocalConstant {
            value: ChoiceValue::Integer(42),
            source_module: "test_module".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };
        
        let constant2 = LocalConstant {
            value: ChoiceValue::Integer(100),
            source_module: "test_module".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };

        cache.insert("key1".to_string(), constant1);
        cache.insert("key2".to_string(), constant2);

        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_some());
        assert_eq!(cache.stats().constants_discovered, 2);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache = LRUConstantCache::new(2);
        
        let constant1 = LocalConstant {
            value: ChoiceValue::Integer(1),
            source_module: "test".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };
        
        let constant2 = LocalConstant {
            value: ChoiceValue::Integer(2),
            source_module: "test".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };
        
        let constant3 = LocalConstant {
            value: ChoiceValue::Integer(3),
            source_module: "test".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };

        cache.insert("key1".to_string(), constant1);
        cache.insert("key2".to_string(), constant2);
        cache.insert("key3".to_string(), constant3);

        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_lru_cache_access_order() {
        let mut cache = LRUConstantCache::new(2);
        
        let constant1 = LocalConstant {
            value: ChoiceValue::Integer(1),
            source_module: "test".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };
        
        let constant2 = LocalConstant {
            value: ChoiceValue::Integer(2),
            source_module: "test".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };

        cache.insert("key1".to_string(), constant1);
        cache.insert("key2".to_string(), constant2);
        
        cache.get("key1");
        
        let constant3 = LocalConstant {
            value: ChoiceValue::Integer(3),
            source_module: "test".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 1.0,
        };
        cache.insert("key3".to_string(), constant3);

        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_none());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_advanced_constant_injection_probability_calculation() {
        let provider = AdvancedConstantInjectionProvider::new();
        
        let int_constraints = Constraints::Integer(IntegerConstraints::new(0, 100));
        let float_constraints = Constraints::Float(FloatConstraints { min: 0.0, max: 1.0, allow_nan: false, smallest_nonzero_magnitude: 1e-10 });
        let string_constraints = Constraints::String(StringConstraints { min_size: 0, max_size: 10, intervals: vec![] });
        let bytes_constraints = Constraints::Bytes(BytesConstraints { min_size: 0, max_size: 10 });
        let bool_constraints = Constraints::Boolean(crate::choice::BooleanConstraints { probability: Some(0.5) });

        assert_eq!((provider.injection_probability_calculator)(&ChoiceType::Integer, &int_constraints), 0.15);
        assert_eq!((provider.injection_probability_calculator)(&ChoiceType::Float, &float_constraints), 0.20);
        assert_eq!((provider.injection_probability_calculator)(&ChoiceType::String, &string_constraints), 0.10);
        assert_eq!((provider.injection_probability_calculator)(&ChoiceType::Bytes, &bytes_constraints), 0.08);
        assert_eq!((provider.injection_probability_calculator)(&ChoiceType::Boolean, &bool_constraints), 0.05);
    }

    #[test]
    fn test_constant_constraint_matching() {
        let provider = AdvancedConstantInjectionProvider::new();
        
        let int_value = ChoiceValue::Integer(50);
        let int_constraints = Constraints::Integer(IntegerConstraints::new(0, 100));
        let int_constraints_out_of_range = Constraints::Integer(IntegerConstraints::new(200, 300));
        
        assert!(provider.constant_matches_type_and_constraints(&int_value, &ChoiceType::Integer, &int_constraints));
        assert!(!provider.constant_matches_type_and_constraints(&int_value, &ChoiceType::Integer, &int_constraints_out_of_range));
        
        let float_value = ChoiceValue::Float(0.5);
        let float_constraints = Constraints::Float(FloatConstraints { min: 0.0, max: 1.0, allow_nan: false, smallest_nonzero_magnitude: 1e-10 });
        let float_constraints_out_of_range = Constraints::Float(FloatConstraints { min: 2.0, max: 3.0, allow_nan: false, smallest_nonzero_magnitude: 1e-10 });
        
        assert!(provider.constant_matches_type_and_constraints(&float_value, &ChoiceType::Float, &float_constraints));
        assert!(!provider.constant_matches_type_and_constraints(&float_value, &ChoiceType::Float, &float_constraints_out_of_range));
        
        let nan_value = ChoiceValue::Float(f64::NAN);
        let float_constraints_allow_nan = Constraints::Float(FloatConstraints { min: 0.0, max: 1.0, allow_nan: true, smallest_nonzero_magnitude: 1e-10 });
        let float_constraints_no_nan = Constraints::Float(FloatConstraints { min: 0.0, max: 1.0, allow_nan: false, smallest_nonzero_magnitude: 1e-10 });
        
        assert!(provider.constant_matches_type_and_constraints(&nan_value, &ChoiceType::Float, &float_constraints_allow_nan));
        assert!(!provider.constant_matches_type_and_constraints(&nan_value, &ChoiceType::Float, &float_constraints_no_nan));
    }

    #[test]
    fn test_advanced_selection_strategy() {
        let provider = AdvancedConstantInjectionProvider::new().with_advanced_selection_strategy();
        
        let old_time = Instant::now() - Duration::from_secs(10);
        let recent_time = Instant::now();
        
        let constants = vec![
            LocalConstant {
                value: ChoiceValue::Integer(1),
                source_module: "test".to_string(),
                discovery_count: 1,
                last_used: old_time,
                selection_weight: 1.0,
            },
            LocalConstant {
                value: ChoiceValue::Integer(2),
                source_module: "test".to_string(),
                discovery_count: 10,
                last_used: recent_time,
                selection_weight: 2.0,
            },
        ];
        
        let constraints = Constraints::Integer(IntegerConstraints::new(0, 100));
        let selected = (provider.constant_selection_strategy)(&constants, &constraints);
        
        assert_eq!(selected, Some(ChoiceValue::Integer(2)));
    }

    #[test]
    fn test_global_constant_fallback() {
        let provider = AdvancedConstantInjectionProvider::new();
        
        let int_constraints = Constraints::Integer(IntegerConstraints::new(-1000, 1000));
        let global_constant = provider.select_global_constant(&ChoiceType::Integer, &int_constraints);
        
        assert!(global_constant.is_some());
        if let Some(ChoiceValue::Integer(val)) = global_constant {
            assert!(val >= -1000 && val <= 1000);
        }
        
        let float_constraints = Constraints::Float(FloatConstraints { min: -1.0, max: 1.0, allow_nan: true, smallest_nonzero_magnitude: 1e-10 });
        let global_float = provider.select_global_constant(&ChoiceType::Float, &float_constraints);
        
        assert!(global_float.is_some());
    }

    #[test] 
    fn test_performance_metrics_collection() {
        let mut provider = AdvancedConstantInjectionProvider::new();
        
        let constraints = Constraints::Integer(IntegerConstraints::new(0, 100));
        provider.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints);
        
        let metrics = provider.get_performance_metrics();
        assert!(metrics.contains_key("constant_selection"));
        assert!(metrics["constant_selection"] > Duration::from_nanos(0));
    }

    #[test]
    fn test_discovery_stats_tracking() {
        let mut provider = AdvancedConstantInjectionProvider::new();
        
        let initial_stats = provider.get_discovery_stats();
        assert_eq!(initial_stats.injection_attempts, 0);
        assert_eq!(initial_stats.successful_injections, 0);
        
        let constraints = Constraints::Integer(IntegerConstraints::new(0, 100));
        
        for _ in 0..10 {
            provider.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints);
        }
        
        let updated_stats = provider.get_discovery_stats();
        assert!(updated_stats.injection_attempts > 0);
    }

    #[test]
    fn test_concurrent_cache_access() {
        let provider = Arc::new(Mutex::new(AdvancedConstantInjectionProvider::new()));
        let mut handles = vec![];
        
        for i in 0..10 {
            let provider_clone = Arc::clone(&provider);
            let handle = thread::spawn(move || {
                let mut p = provider_clone.lock().unwrap();
                let constraints = Constraints::Integer(IntegerConstraints::new(0, 100));
                p.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints)
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_stats = provider.lock().unwrap().get_discovery_stats();
        assert!(final_stats.injection_attempts > 0);
    }

    #[test]
    fn test_cache_size_limits() {
        let mut provider = AdvancedConstantInjectionProvider::new();
        
        {
            let mut cache = provider.local_constant_cache.lock().unwrap();
            for i in 0..1500 {
                let constant = LocalConstant {
                    value: ChoiceValue::Integer(i),
                    source_module: "test".to_string(),
                    discovery_count: 1,
                    last_used: Instant::now(),
                    selection_weight: 1.0,
                };
                cache.insert(format!("key_{}", i), constant);
            }
        }
        
        assert_eq!(provider.cache_size(), 1000);
    }

    #[cfg(feature = "python")]
    #[pyo3_asyncio::tokio::test]
    async fn test_python_constant_discovery() -> PyResult<()> {
        Python::with_gil(|py| {
            let mut provider = AdvancedConstantInjectionProvider::new();
            
            let test_module = PyModule::from_code(
                py,
                r#"
# Test constants for discovery
TEST_INT = 42
TEST_FLOAT = 3.14159
TEST_STRING = "hello world"
TEST_BOOL = True
TEST_LIST = [1, 2, 3]
TEST_DICT = {"key": "value"}

class TestClass:
    CLASS_CONSTANT = 100
"#,
                "test_module.py",
                "test_module",
            )?;
            
            let discovered_count = provider.discover_local_constants(test_module)?;
            assert!(discovered_count > 0);
            
            let stats = provider.get_discovery_stats();
            assert_eq!(stats.modules_scanned, 1);
            assert!(stats.constants_discovered > 0);
            
            Ok(())
        })
    }

    #[cfg(feature = "python")]
    #[pyo3_asyncio::tokio::test]
    async fn test_constant_injection_with_discovered_values() -> PyResult<()> {
        Python::with_gil(|py| {
            let mut provider = AdvancedConstantInjectionProvider::new();
            
            let test_module = PyModule::from_code(
                py,
                r#"
MAGIC_NUMBER = 12345
EDGE_CASE_VALUE = 2147483647
SPECIAL_FLOAT = 2.718281828
"#,
                "constants_module.py",
                "constants_module",
            )?;
            
            provider.discover_local_constants(test_module)?;
            
            let mut injection_count = 0;
            let constraints = Constraints::Integer(IntegerConstraints::new(0, i128::MAX));
            
            for _ in 0..1000 {
                if let Some(ChoiceValue::Integer(val)) = provider.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints) {
                    if val == 12345 || val == 2147483647 {
                        injection_count += 1;
                    }
                }
            }
            
            assert!(injection_count > 0, "Should inject discovered constants");
            
            let stats = provider.get_discovery_stats();
            assert!(stats.successful_injections > 0);
            
            Ok(())
        })
    }

    #[test]
    fn test_mixed_local_and_global_constant_selection() {
        let mut provider = AdvancedConstantInjectionProvider::new();
        
        {
            let mut cache = provider.local_constant_cache.lock().unwrap();
            let local_constant = LocalConstant {
                value: ChoiceValue::Integer(99999),
                source_module: "test".to_string(),
                discovery_count: 5,
                last_used: Instant::now(),
                selection_weight: 2.0,
            };
            cache.insert("unique_local".to_string(), local_constant);
        }
        
        let constraints = Constraints::Integer(IntegerConstraints::new(0, 100000));
        let mut local_injections = 0;
        let mut global_injections = 0;
        
        for _ in 0..1000 {
            if let Some(ChoiceValue::Integer(val)) = provider.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints) {
                if val == 99999 {
                    local_injections += 1;
                } else {
                    global_injections += 1;
                }
            }
        }
        
        assert!(local_injections > 0, "Should inject local constants");
        assert!(global_injections > 0, "Should inject global constants when no locals match");
    }

    #[test]
    fn test_provider_integration_with_primitive_provider_trait() {
        let mut provider = AdvancedConstantInjectionProvider::new();
        
        assert_eq!(provider.lifetime(), ProviderLifetime::TestLifetime);
        
        let int_result = provider.generate_integer(0, 100);
        assert!(int_result.is_ok());
        
        let bool_result = provider.generate_boolean(Some(0.5));
        assert!(bool_result.is_ok());
        
        let float_result = provider.generate_float(0.0, 1.0, false, 1e-10);
        assert!(float_result.is_ok());
        
        let string_result = provider.generate_string(0, 10, vec![]);
        assert!(string_result.is_ok());
        
        let bytes_result = provider.generate_bytes(0, 10);
        assert!(bytes_result.is_ok());
        
        let span_start_result = provider.span_start("test_span".to_string());
        assert!(span_start_result.is_ok());
        
        let span_end_result = provider.span_end(true);
        assert!(span_end_result.is_ok());
    }

    #[test]
    fn test_complex_constraint_filtering() {
        let provider = AdvancedConstantInjectionProvider::new();
        
        let tight_int_constraints = Constraints::Integer(IntegerConstraints::new(50, 60));
        let wide_int_constraints = Constraints::Integer(IntegerConstraints::new(-1000000, 1000000));
        
        let int_value_in_range = ChoiceValue::Integer(55);
        let int_value_out_of_range = ChoiceValue::Integer(1000);
        
        assert!(provider.constant_matches_type_and_constraints(&int_value_in_range, &ChoiceType::Integer, &tight_int_constraints));
        assert!(!provider.constant_matches_type_and_constraints(&int_value_out_of_range, &ChoiceType::Integer, &tight_int_constraints));
        assert!(provider.constant_matches_type_and_constraints(&int_value_out_of_range, &ChoiceType::Integer, &wide_int_constraints));
        
        let special_float_constraints = Constraints::Float(FloatConstraints { 
            min: -1.0, 
            max: 1.0, 
            allow_nan: true, 
            smallest_nonzero_magnitude: 1e-100 
        });
        
        let nan_value = ChoiceValue::Float(f64::NAN);
        let inf_value = ChoiceValue::Float(f64::INFINITY);
        let normal_value = ChoiceValue::Float(0.5);
        
        assert!(provider.constant_matches_type_and_constraints(&nan_value, &ChoiceType::Float, &special_float_constraints));
        assert!(!provider.constant_matches_type_and_constraints(&inf_value, &ChoiceType::Float, &special_float_constraints));
        assert!(provider.constant_matches_type_and_constraints(&normal_value, &ChoiceType::Float, &special_float_constraints));
    }

    #[test]
    fn test_advanced_selection_with_multiple_scoring_factors() {
        let provider = AdvancedConstantInjectionProvider::new().with_advanced_selection_strategy();
        
        let old_frequent_constant = LocalConstant {
            value: ChoiceValue::Integer(100),
            source_module: "legacy_module".to_string(),
            discovery_count: 1000,
            last_used: Instant::now() - Duration::from_secs(3600),
            selection_weight: 1.0,
        };
        
        let recent_rare_constant = LocalConstant {
            value: ChoiceValue::Integer(200),
            source_module: "new_module".to_string(),
            discovery_count: 1,
            last_used: Instant::now(),
            selection_weight: 3.0,
        };
        
        let balanced_constant = LocalConstant {
            value: ChoiceValue::Integer(300),
            source_module: "balanced_module".to_string(),
            discovery_count: 50,
            last_used: Instant::now() - Duration::from_secs(300),
            selection_weight: 2.0,
        };
        
        let constants = vec![old_frequent_constant, recent_rare_constant, balanced_constant];
        let constraints = Constraints::Integer(IntegerConstraints::new(0, 1000));
        
        let selected = (provider.constant_selection_strategy)(&constants, &constraints);
        assert!(selected.is_some());
        
        if let Some(ChoiceValue::Integer(val)) = selected {
            assert!(vec![100, 200, 300].contains(&val));
        }
    }

    #[test]
    fn test_discovery_performance_under_load() {
        let mut provider = AdvancedConstantInjectionProvider::new();
        
        let start_time = Instant::now();
        
        for _ in 0..10000 {
            let constraints = Constraints::Integer(IntegerConstraints::new(0, 1000));
            provider.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints);
        }
        
        let total_time = start_time.elapsed();
        let metrics = provider.get_performance_metrics();
        
        assert!(total_time < Duration::from_secs(1), "Should complete 10k operations within 1 second");
        assert!(metrics.contains_key("constant_selection"));
        
        let stats = provider.get_discovery_stats();
        assert!(stats.injection_attempts > 0);
    }

    #[test]
    fn test_comprehensive_capability_integration() {
        let mut provider = AdvancedConstantInjectionProvider::new().with_advanced_selection_strategy();
        
        {
            let mut cache = provider.local_constant_cache.lock().unwrap();
            for i in 0..100 {
                let constant = LocalConstant {
                    value: ChoiceValue::Integer(i * 100),
                    source_module: format!("module_{}", i % 10),
                    discovery_count: (i % 20) + 1,
                    last_used: Instant::now() - Duration::from_secs(i % 60),
                    selection_weight: (i % 5 + 1) as f64,
                };
                cache.insert(format!("constant_{}", i), constant);
            }
        }
        
        let constraints = Constraints::Integer(IntegerConstraints::new(0, 10000));
        let mut injection_results = HashMap::new();
        
        for _ in 0..5000 {
            if let Some(ChoiceValue::Integer(val)) = provider.maybe_draw_advanced_constant(ChoiceType::Integer, &constraints) {
                *injection_results.entry(val).or_insert(0) += 1;
            }
        }
        
        let stats = provider.get_discovery_stats();
        let metrics = provider.get_performance_metrics();
        
        assert!(stats.injection_attempts > 0);
        assert!(stats.successful_injections > 0);
        assert!(injection_results.len() > 0);
        assert!(metrics.contains_key("constant_selection"));
        
        let injection_success_rate = stats.successful_injections as f64 / stats.injection_attempts as f64;
        assert!(injection_success_rate > 0.05, "Should have reasonable injection success rate");
        
        assert_eq!(provider.cache_size(), 100);
    }
}