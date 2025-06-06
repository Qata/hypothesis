use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::slice;

use conjecture::data::{DataSource, Status};
use conjecture::database::{BoxedDatabase, DirectoryDatabase, NoDatabase};
use conjecture::distributions::{self, Repeat};
use conjecture::engine::{Engine, Phase};
use conjecture::floats;

// Error codes matching Swift
pub const CONJECTURE_SUCCESS: i32 = 0;
pub const CONJECTURE_ERROR_NULL_HANDLE: i32 = -1;
pub const CONJECTURE_ERROR_INDEX_OUT_OF_BOUNDS: i32 = -2;
pub const CONJECTURE_ERROR_INTERNAL: i32 = -3;
pub const CONJECTURE_ERROR_INVALID_STRING: i32 = -4;
pub const CONJECTURE_ERROR_DATA_OVERFLOW: i32 = -5;
pub const CONJECTURE_ERROR_INVALID_PHASE: i32 = -6;

pub const CONJECTURE_PHASE_SHRINK: isize = 0;

// Version information
const VERSION_MAJOR: u16 = 0;
const VERSION_MINOR: u8 = 1;
const VERSION_PATCH: u8 = 0;

// C-compatible phase representation
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CPhase {
    Shrink = CONJECTURE_PHASE_SHRINK,
}

// Convert internal Rust Phase to C Phase
impl From<Phase> for CPhase {
    fn from(phase: Phase) -> Self {
        match phase {
            Phase::Shrink => CPhase::Shrink,
        }
    }
}

// Convert C Phase to internal Rust Phase
impl TryFrom<CPhase> for Phase {
    type Error = i32;
    
    fn try_from(c_phase: CPhase) -> Result<Self, Self::Error> {
        match c_phase {
            CPhase::Shrink => Ok(Phase::Shrink),
        }
    }
}

// Wrapper types to make Rust types FFI-safe
#[repr(C)]
pub struct EngineWrapper {
    engine: Engine,
    pending_source: Option<DataSource>,
    interesting_examples: Vec<conjecture::data::TestResult>,
}

struct DataSourceWrapper {
    source: Option<DataSource>,
}

struct IntegersWrapper {
    bitlengths: distributions::Sampler,
}

struct BoundedIntegersWrapper {
    max_value: u64,
}

struct RepeatValuesWrapper {
    repeat: Repeat,
}

struct FloatsWrapper {
    // No state needed - uses static functions
}

// Helper macro for null handle checks
macro_rules! check_handle {
    ($handle:expr) => {
        match unsafe { $handle.as_ref() } {
            Some(h) => h,
            None => return CONJECTURE_ERROR_NULL_HANDLE,
        }
    };
}

macro_rules! check_handle_mut {
    ($handle:expr) => {
        match unsafe { $handle.as_mut() } {
            Some(h) => h,
            None => return CONJECTURE_ERROR_NULL_HANDLE,
        }
    };
}

// Engine FFI functions
#[no_mangle]
pub extern "C" fn conjecture_engine_new(
    name: *const c_char,
    database_path: *const c_char,
    seed: u64,
    max_examples: u64,
    phases_ptr: *const CPhase,
    phases_len: usize,
    result: *mut *mut c_void,
) -> i32 {
    // Parse name
    let name = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return CONJECTURE_ERROR_INVALID_STRING,
    };
    
    // Parse database path
    let db_path = if database_path.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(database_path) }.to_str() {
            Ok(s) => Some(s.to_string()),
            Err(_) => return CONJECTURE_ERROR_INVALID_STRING,
        }
    };
    
    // Parse phases
    let phases = if phases_ptr.is_null() || phases_len == 0 {
        Phase::all()
    } else {
        let phase_slice = unsafe { slice::from_raw_parts(phases_ptr, phases_len) };
        let mut rust_phases = Vec::new();
        for &c_phase in phase_slice {
            match Phase::try_from(c_phase) {
                Ok(phase) => rust_phases.push(phase),
                Err(_) => return CONJECTURE_ERROR_INVALID_PHASE,
            }
        }
        rust_phases
    };
    
    // Create database
    let db: BoxedDatabase = match db_path {
        Some(path) => Box::new(DirectoryDatabase::new(path)),
        None => Box::new(NoDatabase),
    };
    
    // Create engine
    let seed_array: [u32; 2] = [seed as u32, (seed >> 32) as u32];
    let engine = Engine::new(name, max_examples, phases, &seed_array, db);
    
    let wrapper = Box::new(EngineWrapper {
        engine,
        pending_source: None,
        interesting_examples: Vec::new(),
    });
    
    unsafe {
        *result = Box::into_raw(wrapper) as *mut c_void;
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut EngineWrapper);
        }
    }
}

#[no_mangle]
pub extern "C" fn conjecture_engine_new_source(
    handle: *mut c_void,
    result: *mut *mut c_void,
) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut EngineWrapper);
    
    match wrapper.engine.next_source() {
        Some(source) => {
            wrapper.pending_source = Some(source.clone());
            let ds_wrapper = Box::new(DataSourceWrapper {
                source: Some(source),
            });
            unsafe {
                *result = Box::into_raw(ds_wrapper) as *mut c_void;
            }
        }
        None => {
            wrapper.interesting_examples = wrapper.engine.list_minimized_examples();
            unsafe {
                *result = ptr::null_mut();
            }
        }
    }
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_count_failing_examples(
    handle: *mut c_void,
    result: *mut usize,
) -> i32 {
    let wrapper = check_handle!(handle as *mut EngineWrapper);
    unsafe {
        *result = wrapper.interesting_examples.len();
    }
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_failing_example(
    handle: *mut c_void,
    index: usize,
    result: *mut *mut c_void,
) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut EngineWrapper);
    
    if index >= wrapper.interesting_examples.len() {
        return CONJECTURE_ERROR_INDEX_OUT_OF_BOUNDS;
    }
    
    let source = DataSource::from_vec(wrapper.interesting_examples[index].record.clone());
    wrapper.pending_source = Some(source.clone());
    
    let ds_wrapper = Box::new(DataSourceWrapper {
        source: Some(source),
    });
    
    unsafe {
        *result = Box::into_raw(ds_wrapper) as *mut c_void;
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_was_unsatisfiable(
    handle: *mut c_void,
    result: *mut bool,
) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut EngineWrapper);
    unsafe {
        *result = wrapper.engine.was_unsatisfiable();
    }
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_finish_overflow(
    engine_handle: *mut c_void,
    ds_handle: *mut c_void,
) -> i32 {
    let engine_wrapper = check_handle_mut!(engine_handle as *mut EngineWrapper);
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(source) = ds_wrapper.source.take() {
        engine_wrapper.engine.mark_finished(source, Status::Overflow);
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_finish_valid(
    engine_handle: *mut c_void,
    ds_handle: *mut c_void,
) -> i32 {
    let engine_wrapper = check_handle_mut!(engine_handle as *mut EngineWrapper);
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(source) = ds_wrapper.source.take() {
        engine_wrapper.engine.mark_finished(source, Status::Valid);
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_finish_invalid(
    engine_handle: *mut c_void,
    ds_handle: *mut c_void,
) -> i32 {
    let engine_wrapper = check_handle_mut!(engine_handle as *mut EngineWrapper);
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(source) = ds_wrapper.source.take() {
        engine_wrapper.engine.mark_finished(source, Status::Invalid);
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_engine_finish_interesting(
    engine_handle: *mut c_void,
    ds_handle: *mut c_void,
    label: u64,
) -> i32 {
    let engine_wrapper = check_handle_mut!(engine_handle as *mut EngineWrapper);
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(source) = ds_wrapper.source.take() {
        engine_wrapper.engine.mark_finished(source, Status::Interesting(label));
    }
    
    CONJECTURE_SUCCESS
}

// DataSource FFI functions
#[no_mangle]
pub extern "C" fn conjecture_data_source_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut DataSourceWrapper);
        }
    }
}

#[no_mangle]
pub extern "C" fn conjecture_data_source_start_draw(handle: *mut c_void) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = wrapper.source {
        source.start_draw();
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_data_source_stop_draw(handle: *mut c_void) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = wrapper.source {
        source.stop_draw();
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_data_source_bits(
    handle: *mut c_void,
    n_bits: u64,
    result: *mut u64,
) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = wrapper.source {
        match source.bits(n_bits) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                CONJECTURE_SUCCESS
            }
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

#[no_mangle]
pub extern "C" fn conjecture_data_source_write(
    handle: *mut c_void,
    value: u64,
) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = wrapper.source {
        match source.write(value) {
            Ok(_) => CONJECTURE_SUCCESS,
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

// Integers FFI functions
#[no_mangle]
pub extern "C" fn conjecture_integers_new(result: *mut *mut c_void) -> i32 {
    let wrapper = Box::new(IntegersWrapper {
        bitlengths: distributions::good_bitlengths(),
    });
    
    unsafe {
        *result = Box::into_raw(wrapper) as *mut c_void;
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_integers_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut IntegersWrapper);
        }
    }
}

#[no_mangle]
pub extern "C" fn conjecture_integers_provide(
    integers_handle: *mut c_void,
    ds_handle: *mut c_void,
    result: *mut i64,
) -> i32 {
    let integers_wrapper = check_handle_mut!(integers_handle as *mut IntegersWrapper);
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = ds_wrapper.source {
        match distributions::integer_from_bitlengths(source, &integers_wrapper.bitlengths) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                CONJECTURE_SUCCESS
            }
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

// RepeatValues FFI functions
#[no_mangle]
pub extern "C" fn conjecture_repeat_values_new(
    min_count: u64,
    max_count: u64,
    expected_count: f64,
    result: *mut *mut c_void,
) -> i32 {
    let wrapper = Box::new(RepeatValuesWrapper {
        repeat: Repeat::new(min_count, max_count, expected_count),
    });
    
    unsafe {
        *result = Box::into_raw(wrapper) as *mut c_void;
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_repeat_values_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut RepeatValuesWrapper);
        }
    }
}

#[no_mangle]
pub extern "C" fn conjecture_repeat_values_should_continue(
    repeat_handle: *mut c_void,
    ds_handle: *mut c_void,
    result: *mut bool,
) -> i32 {
    let repeat_wrapper = check_handle_mut!(repeat_handle as *mut RepeatValuesWrapper);
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = ds_wrapper.source {
        match repeat_wrapper.repeat.should_continue(source) {
            Ok(should_continue) => {
                unsafe {
                    *result = should_continue;
                }
                CONJECTURE_SUCCESS
            }
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

#[no_mangle]
pub extern "C" fn conjecture_repeat_values_reject(handle: *mut c_void) -> i32 {
    let wrapper = check_handle_mut!(handle as *mut RepeatValuesWrapper);
    wrapper.repeat.reject();
    CONJECTURE_SUCCESS
}

// BoundedIntegers FFI functions
#[no_mangle]
pub extern "C" fn conjecture_bounded_integers_new(
    max_value: u64,
    result: *mut *mut c_void,
) -> i32 {
    let wrapper = Box::new(BoundedIntegersWrapper { max_value });
    
    unsafe {
        *result = Box::into_raw(wrapper) as *mut c_void;
    }
    
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_bounded_integers_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut BoundedIntegersWrapper);
        }
    }
}

#[no_mangle]
pub extern "C" fn conjecture_bounded_integers_provide(
    bounded_handle: *mut c_void,
    ds_handle: *mut c_void,
    result: *mut u64,
) -> i32 {
    let bounded_wrapper = check_handle!(bounded_handle as *mut BoundedIntegersWrapper);
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = ds_wrapper.source {
        match distributions::bounded_int(source, bounded_wrapper.max_value) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                CONJECTURE_SUCCESS
            }
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

// Float generation FFI functions
#[no_mangle]
pub extern "C" fn conjecture_floats_new(result: *mut *mut c_void) -> i32 {
    if result.is_null() {
        return CONJECTURE_ERROR_NULL_HANDLE;
    }
    
    let wrapper = Box::new(FloatsWrapper {});
    unsafe {
        *result = Box::into_raw(wrapper) as *mut c_void;
    }
    CONJECTURE_SUCCESS
}

#[no_mangle]
pub extern "C" fn conjecture_floats_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            let _wrapper = Box::from_raw(handle as *mut FloatsWrapper);
        }
    }
}

#[no_mangle]
pub extern "C" fn conjecture_floats_provide_bounded(
    _floats_handle: *mut c_void,
    ds_handle: *mut c_void,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
    result: *mut f64,
) -> i32 {
    if result.is_null() {
        return CONJECTURE_ERROR_NULL_HANDLE;
    }
    
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = ds_wrapper.source {
        match distributions::float_with_bounds(source, min_value, max_value, allow_nan, allow_infinity) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                CONJECTURE_SUCCESS
            }
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

#[no_mangle]
pub extern "C" fn conjecture_floats_provide_any(
    _floats_handle: *mut c_void,
    ds_handle: *mut c_void,
    result: *mut f64,
) -> i32 {
    if result.is_null() {
        return CONJECTURE_ERROR_NULL_HANDLE;
    }
    
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = ds_wrapper.source {
        match distributions::any_float(source) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                CONJECTURE_SUCCESS
            }
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

#[no_mangle]
pub extern "C" fn conjecture_floats_provide_uniform(
    _floats_handle: *mut c_void,
    ds_handle: *mut c_void,
    min_value: f64,
    max_value: f64,
    result: *mut f64,
) -> i32 {
    if result.is_null() {
        return CONJECTURE_ERROR_NULL_HANDLE;
    }
    
    let ds_wrapper = check_handle_mut!(ds_handle as *mut DataSourceWrapper);
    
    if let Some(ref mut source) = ds_wrapper.source {
        match distributions::uniform_float(source, min_value, max_value) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                CONJECTURE_SUCCESS
            }
            Err(_) => CONJECTURE_ERROR_DATA_OVERFLOW,
        }
    } else {
        CONJECTURE_ERROR_INTERNAL
    }
}

// Version function
#[no_mangle]
pub extern "C" fn conjecture_ffi_version() -> u32 {
    ((VERSION_MAJOR as u32) << 16) | ((VERSION_MINOR as u32) << 8) | (VERSION_PATCH as u32)
}
