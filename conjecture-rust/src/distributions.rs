// General distribution functions and utilities.
// This module contains probability distributions and repetition
// control that are used across different data types.

use crate::data::{DataSource, FailedDraw};

use std::u64::MAX as MAX64;

pub fn weighted(source: &mut DataSource, probability: f64) -> Result<bool, FailedDraw> {
    // TODO: Less bit-hungry implementation.

    let truthy = (probability * (u64::max_value() as f64 + 1.0)).floor() as u64;
    let probe = source.bits(64)?;
    Ok(match (truthy, probe) {
        (0, _) => false,
        (MAX64, _) => true,
        (_, 0) => false,
        (_, 1) => true,
        _ => probe >= MAX64 - truthy,
    })
}

#[derive(Debug, Clone)]
pub struct Repeat {
    min_count: u64,
    max_count: u64,
    p_continue: f64,

    current_count: u64,
}

impl Repeat {
    pub fn new(min_count: u64, max_count: u64, expected_count: f64) -> Repeat {
        Repeat {
            min_count,
            max_count,
            p_continue: 1.0 - 1.0 / (1.0 + expected_count),
            current_count: 0,
        }
    }

    pub fn reject(&mut self) {
        assert!(self.current_count > 0);
        self.current_count -= 1;
    }

    pub fn should_continue(&mut self, source: &mut DataSource) -> Result<bool, FailedDraw> {
        if self.min_count == self.max_count {
            if self.current_count < self.max_count {
                self.current_count += 1;
                return Ok(true);
            } else {
                return Ok(false);
            }
        } else if self.current_count < self.min_count {
            source.write(1)?;
            self.current_count += 1;
            return Ok(true);
        } else if self.current_count >= self.max_count {
            source.write(0)?;
            return Ok(false);
        }

        let result = weighted(source, self.p_continue)?;
        if result {
            self.current_count += 1;
        } else {
        }
        Ok(result)
    }
}