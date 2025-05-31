#!/usr/bin/env bash

cargo install cbindgen
cbindgen --lang c --output include/conjecture.h

rustup target add aarch64-apple-darwin
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-macabi
rustup target add aarch64-apple-ios-sim
rustup target add aarch64-apple-tvos
rustup target add aarch64-apple-tvos-sim
rustup target add aarch64-apple-visionos
rustup target add aarch64-apple-visionos-sim
rustup target add aarch64-apple-watchos
rustup target add aarch64-apple-watchos-sim
rustup target add arm64_32-apple-watchos
rustup target add arm64e-apple-darwin
rustup target add arm64e-apple-ios
rustup target add arm64e-apple-tvos
rustup target add armv7k-apple-watchos
rustup target add armv7s-apple-ios
rustup target add i386-apple-ios
rustup target add i686-apple-darwin
rustup target add x86_64-apple-darwin
rustup target add x86_64-apple-ios
rustup target add x86_64-apple-ios-macabi
rustup target add x86_64-apple-tvos
rustup target add x86_64-apple-watchos-sim
rustup target add x86_64h-apple-darwin
