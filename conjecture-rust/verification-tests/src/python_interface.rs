//! Python interface using subprocess calls

use conjecture::choice::*;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};

/// Python interface for calling Hypothesis choice functions via subprocess
pub struct PythonInterface {
    python_script_path: String,
}

#[derive(Serialize, Deserialize)]
struct PythonRequest {
    value: Option<serde_json::Value>,
    constraints: Option<serde_json::Value>,
    index: Option<u128>,
    choice_type: Option<String>,
    value_a: Option<serde_json::Value>,
    value_b: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct PythonResponse {
    result: Option<serde_json::Value>,
    error: Option<String>,
}

impl PythonInterface {
    /// Create new Python interface
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let script_path = std::env::current_dir()?
            .join("verify_python.py")
            .to_string_lossy()
            .to_string();
        
        // Test that Python script exists and is executable
        if !std::path::Path::new(&script_path).exists() {
            return Err(format!("Python script not found: {}", script_path).into());
        }
        
        Ok(Self {
            python_script_path: script_path,
        })
    }

    /// Call Python's choice_to_index function
    pub fn choice_to_index(&self, value: &ChoiceValue, constraints: &Constraints) -> Result<u128, Box<dyn std::error::Error>> {
        let value_json = self.choice_value_to_json(value)?;
        let constraints_json = self.constraints_to_json(constraints)?;
        
        let request = PythonRequest {
            value: Some(value_json),
            constraints: Some(constraints_json),
            index: None,
            choice_type: None,
            value_a: None,
            value_b: None,
        };
        
        let response = self.call_python("choice_to_index", &request)?;
        
        match response.result {
            Some(serde_json::Value::Number(n)) => {
                if let Some(n) = n.as_u64() {
                    Ok(n as u128)
                } else {
                    Err("Invalid numeric response from Python".into())
                }
            }
            _ => Err(format!("Unexpected response format: {:?}", response).into()),
        }
    }

    /// Call Python's choice_from_index function
    pub fn choice_from_index(
        &self,
        index: u128,
        choice_type: &str,
        constraints: &Constraints
    ) -> Result<ChoiceValue, Box<dyn std::error::Error>> {
        let constraints_json = self.constraints_to_json(constraints)?;
        
        let request = PythonRequest {
            value: None,
            constraints: Some(constraints_json),
            index: Some(index),
            choice_type: Some(choice_type.to_string()),
            value_a: None,
            value_b: None,
        };
        
        let response = self.call_python("choice_from_index", &request)?;
        
        match response.result {
            Some(value_json) => self.json_to_choice_value(&value_json, choice_type),
            None => Err("No result from Python choice_from_index".into()),
        }
    }

    /// Call Python's choice_equal function
    pub fn choice_equal(&self, a: &ChoiceValue, b: &ChoiceValue) -> Result<bool, Box<dyn std::error::Error>> {
        let value_a_json = self.choice_value_to_json(a)?;
        let value_b_json = self.choice_value_to_json(b)?;
        
        let request = PythonRequest {
            value: None,
            constraints: None,
            index: None,
            choice_type: None,
            value_a: Some(value_a_json),
            value_b: Some(value_b_json),
        };
        
        let response = self.call_python("choice_equal", &request)?;
        
        match response.result {
            Some(serde_json::Value::Bool(b)) => Ok(b),
            _ => Err(format!("Unexpected response format: {:?}", response).into()),
        }
    }

    /// Call Python subprocess with given command and request
    fn call_python(&self, command: &str, request: &PythonRequest) -> Result<PythonResponse, Box<dyn std::error::Error>> {
        let mut child = Command::new("python3")
            .arg(&self.python_script_path)
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Send JSON input to Python
        if let Some(stdin) = child.stdin.as_mut() {
            let json_input = serde_json::to_string(request)?;
            stdin.write_all(json_input.as_bytes())?;
        }

        let output = child.wait_with_output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Python script failed: {}", stderr).into());
        }

        let stdout = String::from_utf8(output.stdout)?;
        let response: PythonResponse = serde_json::from_str(&stdout)?;
        
        if let Some(error) = &response.error {
            return Err(format!("Python error: {}", error).into());
        }
        
        Ok(response)
    }

    /// Convert ChoiceValue to JSON
    fn choice_value_to_json(&self, value: &ChoiceValue) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        match value {
            ChoiceValue::Integer(val) => Ok(serde_json::Value::Number((*val).into())),
            ChoiceValue::Boolean(val) => Ok(serde_json::Value::Bool(*val)),
            ChoiceValue::Float(val) => Ok(serde_json::to_value(val)?),
            ChoiceValue::String(val) => Ok(serde_json::Value::String(val.clone())),
            ChoiceValue::Bytes(val) => Ok(serde_json::to_value(val)?),
        }
    }

    /// Convert Constraints to JSON
    fn constraints_to_json(&self, constraints: &Constraints) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        match constraints {
            Constraints::Integer(c) => {
                let mut obj = serde_json::Map::new();
                obj.insert("min_value".to_string(), serde_json::to_value(&c.min_value)?);
                obj.insert("max_value".to_string(), serde_json::to_value(&c.max_value)?);
                obj.insert("weights".to_string(), serde_json::to_value(&c.weights)?);
                obj.insert("shrink_towards".to_string(), serde_json::to_value(&c.shrink_towards.unwrap_or(0))?);
                Ok(serde_json::Value::Object(obj))
            },
            Constraints::Boolean(c) => {
                let mut obj = serde_json::Map::new();
                obj.insert("p".to_string(), serde_json::to_value(c.p)?);
                Ok(serde_json::Value::Object(obj))
            },
            Constraints::Float(c) => {
                let mut obj = serde_json::Map::new();
                obj.insert("min_value".to_string(), serde_json::to_value(c.min_value)?);
                obj.insert("max_value".to_string(), serde_json::to_value(c.max_value)?);
                obj.insert("allow_nan".to_string(), serde_json::to_value(c.allow_nan)?);
                obj.insert("smallest_nonzero_magnitude".to_string(), serde_json::to_value(&c.smallest_nonzero_magnitude)?);
                Ok(serde_json::Value::Object(obj))
            },
            _ => Err("String and Bytes constraints not implemented yet".into()),
        }
    }

    /// Convert JSON to ChoiceValue
    fn json_to_choice_value(&self, json: &serde_json::Value, choice_type: &str) -> Result<ChoiceValue, Box<dyn std::error::Error>> {
        match choice_type {
            "integer" => {
                if let Some(n) = json.as_i64() {
                    Ok(ChoiceValue::Integer(n as i128))
                } else {
                    Err(format!("Invalid integer value: {:?}", json).into())
                }
            },
            "boolean" => {
                if let Some(b) = json.as_bool() {
                    Ok(ChoiceValue::Boolean(b))
                } else {
                    Err(format!("Invalid boolean value: {:?}", json).into())
                }
            },
            "float" => {
                if let Some(f) = json.as_f64() {
                    Ok(ChoiceValue::Float(f))
                } else {
                    Err(format!("Invalid float value: {:?}", json).into())
                }
            },
            "string" => {
                if let Some(s) = json.as_str() {
                    Ok(ChoiceValue::String(s.to_string()))
                } else {
                    Err(format!("Invalid string value: {:?}", json).into())
                }
            },
            "bytes" => {
                if let Ok(bytes) = serde_json::from_value::<Vec<u8>>(json.clone()) {
                    Ok(ChoiceValue::Bytes(bytes))
                } else {
                    Err(format!("Invalid bytes value: {:?}", json).into())
                }
            },
            _ => Err(format!("Unsupported choice type: {}", choice_type).into()),
        }
    }
}