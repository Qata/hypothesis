# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Ruby port of the Hypothesis property-based testing library. The project consists of:

- **Ruby Layer**: Public API for Ruby developers (in `lib/hypothesis/`)
- **Rust Core**: Core testing engine and data generation logic (in `src/`)
- **FFI Bridge**: Rutie bindings that bridge Ruby and Rust
- **Native Extension**: Compiled Rust library that Ruby loads via FFI

The architecture follows a layered approach where Ruby provides the user-facing API while delegating core functionality to a Rust implementation via FFI.

## Common Commands

### Building
- `cargo build --release` - Build the Rust core library
- `cargo build --release --target x86_64-apple-darwin` - Build for specific architecture if needed

### Testing
- `bundle exec rspec` - Run all Ruby tests
- `bundle exec rspec spec/float_spec.rb` - Run specific test file
- `rake test` - Run complete test suite including minitest
- `rake spec` - Run just RSpec tests

### Development Setup
- Use rbenv with Ruby 2.6.10 or later for compatibility with modern Rust toolchain
- `bundle install` - Install Ruby dependencies
- Build Rust library first before running tests

## Architecture

### Core Components

**Engine (`lib/hypothesis/engine.rb`)**: Main test execution engine that:
- Manages test case lifecycle through the Rust core via FFI
- Handles different test outcomes (valid, invalid, interesting, overflow)
- Provides error wrapping and reporting

**TestCase (`lib/hypothesis/testcase.rb`)**: Represents a single test execution with methods for:
- Drawing values from data sources
- Recording/printing draws for debugging
- Managing test assumptions

**FFI Layer (`src/lib.rs`)**: Bridges Ruby and Rust using Rutie:
- `HypothesisCoreEngine`: Wraps Rust engine functionality
- `HypothesisCoreDataSource`: Wraps Rust data source for value generation
- Various core types for integers, floats, repeat values, bounded integers

**Rust Core**: Provides the underlying testing engine via FFI:
- Data source management and bit generation
- Test case shrinking and minimization  
- Database integration for test persistence
- Distribution sampling for various data types

### Data Flow

1. `hypothesis()` function creates an `Engine` instance
2. Engine requests new data sources from Rust core via Rutie FFI
3. TestCase wraps data source and provides Ruby-friendly value generation
4. Test execution results are communicated back to Rust core
5. Engine handles shrinking and example minimization through Rust

### Value Generation

The library provides "Possible" types in `lib/hypothesis/possible.rb` that generate test values:
- `integers`: Unbounded and bounded integer generation
- `floats`: Floating-point number generation with sophisticated lexicographic encoding
- `strings`: String generation with configurable codepoints
- `arrays`: Array generation with configurable elements and sizes
- `hashes`: Hash generation with fixed or variable shapes
- All use the core data source for deterministic, shrinkable generation

## Key Patterns

- All public APIs raise `HypothesisError` variants for different failure modes
- Test execution is managed through a global `World.current_engine` to prevent nesting
- FFI calls return nil on failure, which triggers `DataOverflow` exceptions
- Value generation is lazy and driven by the underlying Rust data source
- Ruby wrapper classes follow the pattern of implementing `provide(source)` method

## Development Notes

- **Don't ask permission for routine bash commands** - just run them as needed for development tasks
- Ruby 2.4.2 has compatibility issues with modern Rust - use Ruby 2.6.10+ 
- Architecture mismatches between Ruby and Rust library will cause segfaults
- The Rust library must be built for the same architecture as the Ruby interpreter
- FFI debugging can be challenging - start with syntax validation and mock testing