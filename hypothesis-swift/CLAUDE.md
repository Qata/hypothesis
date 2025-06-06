# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Swift port of the Hypothesis property-based testing library. The project consists of:

- **Swift Layer**: Public API for Swift developers (in `Sources/Hypothesis/`)
- **Rust Core**: Core testing engine and data generation logic (in `src/`)
- **FFI Bridge**: C bindings that bridge Swift and Rust (generated header at `include/conjecture.h`)
- **XCFramework**: Pre-built binary framework for distribution (`libs/Conjecture.xcframework`)

The architecture follows a layered approach where Swift provides the user-facing API while delegating core functionality to a Rust implementation via FFI.

## Common Commands

### Building
- `make all` - Build for all Apple platforms (macOS, iOS, tvOS, watchOS, visionOS)
- `make macos` - Build specifically for macOS and Mac Catalyst
- `make ios` - Build specifically for iOS device and simulator
- `make framework` - Create the XCFramework from all platform static libraries
- `cargo build --release` - Build just the Rust core library

### Testing
- `swift test` - Run all Swift tests
- `swift test --filter <pattern>` - Run specific tests matching pattern
- `swift test -v` - Run tests with verbose output

### Development Setup
- `./build.sh` - Install all required Rust targets for Apple platforms
- `cbindgen --lang c --output include/conjecture.h` - Regenerate C header from Rust

## Architecture

### Core Components

**Engine (`Sources/Hypothesis/Engine.swift`)**: Main test execution engine that:
- Manages test case lifecycle through the Rust core
- Handles different test outcomes (valid, invalid, interesting, overflow)
- Provides error wrapping and reporting
- Manages the `find` mode for targeted example discovery

**TestCase (`Sources/Hypothesis/TestCase.swift`)**: Represents a single test execution with methods for:
- Drawing values from data sources
- Recording/printing draws for debugging
- Managing test assumptions

**FFI Layer (`Sources/Hypothesis/Conjecture/`)**: Bridges Swift and Rust:
- `CoreEngine`: Wraps Rust engine functionality
- `CoreDataSource`: Wraps Rust data source for value generation
- Various core types for integers, doubles, repeat values

**Rust Core (`src/lib.rs`)**: Provides the underlying testing engine via C FFI:
- Data source management and bit generation
- Test case shrinking and minimization
- Database integration for test persistence
- Distribution sampling for various data types

### Data Flow

1. `hypothesis()` function creates an `Engine` instance
2. Engine requests new data sources from Rust core via FFI
3. TestCase wraps data source and provides Swift-friendly value generation
4. Test execution results are communicated back to Rust core
5. Engine handles shrinking and example minimization through Rust

### Value Generation

The library provides "Possible" types in `Sources/Hypothesis/Possibles/` that generate test values:
- `Integers`: Unbounded integer generation
- `Doubles`: Floating-point number generation  
- `Possibilities`: For choosing from finite sets
- All use the core data source for deterministic, shrinkable generation

## Key Patterns

- All public APIs throw `HypothesisError` variants for different failure modes
- Test execution is managed through a global `World.currentEngine` to prevent nesting
- FFI calls return error codes that are converted to Swift exceptions
- Value generation is lazy and driven by the underlying Rust data source