#![allow(clippy::many_single_char_names)]
#![warn(clippy::cargo, rust_2018_idioms, rust_2018_compatibility)]
pub mod data;
pub mod database;
pub mod distributions;
pub mod engine;
pub mod floats;
pub mod intminimize;
pub mod ints;
pub mod strings;

#[cfg(test)]
mod python_parity_tests;
