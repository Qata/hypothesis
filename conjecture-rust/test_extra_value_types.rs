use conjecture_rust::data::{ExtraInformation, ExtraValue};

fn main() {
    println!("Testing ExtraValue with multiple data types...");
    
    let mut extra = ExtraInformation::new();
    
    // Test different types
    let _ = extra.insert_str("string_key", "Hello World");
    let _ = extra.insert_int("int_key", 42);
    let _ = extra.insert_float("float_key", 3.14);
    let _ = extra.insert_bool("bool_key", true);
    let _ = extra.insert_bytes("bytes_key", vec![65, 66, 67]); // ABC
    let _ = extra.insert_none("none_key");
    
    println!("ExtraInformation display: {}", extra);
    
    // Test getting values back
    if let Some(val) = extra.get("string_key") {
        println!("string_key repr: {}", val.repr());
    }
    if let Some(val) = extra.get("int_key") {
        println!("int_key repr: {}", val.repr());
    }
    if let Some(val) = extra.get("float_key") {
        println!("float_key repr: {}", val.repr());
    }
    if let Some(val) = extra.get("bool_key") {
        println!("bool_key repr: {}", val.repr());
    }
    if let Some(val) = extra.get("bytes_key") {
        println!("bytes_key repr: {}", val.repr());
    }
    if let Some(val) = extra.get("none_key") {
        println!("none_key repr: {}", val.repr());
    }
}