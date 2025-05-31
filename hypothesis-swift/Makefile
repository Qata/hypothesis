.PHONY: all common macos ios tvos watchos visionos

all: macos ios tvos watchos visionos

common:
	@echo "Running common setup..."
	@cargo install cbindgen
	@cbindgen --lang c --output include/conjecture.h
	@mkdir -p libs
	
framework:
	@rm -rf ./libs/Conjecture.xcframework
	@echo "Creating XCFramework from all .a files in ./libs"
	@cmd="xcodebuild -create-xcframework"; \
	for lib in libs/*.a; do \
		echo "Adding $$lib"; \
		codesign --force --sign - --timestamp=none $$lib; \
		cmd="$$cmd -library $$lib -headers ./include/"; \
	done; \
	$$cmd -output ./libs/Conjecture.xcframework
	@codesign --force --sign - --timestamp=none ./libs/Conjecture.xcframework

macos: common
	@rustup target add aarch64-apple-darwin
	@rustup target add x86_64-apple-darwin
	@rustup target add aarch64-apple-ios-macabi
	@rustup target add x86_64-apple-ios-macabi
	@cargo build --release --lib --target aarch64-apple-darwin
	@cargo build --release --lib --target x86_64-apple-darwin
	@cargo build --release --lib --target aarch64-apple-ios-macabi
	@cargo build --release --lib --target x86_64-apple-ios-macabi
	@$(RM) -f libs/libhypothesis_swift_core-macos.a
	@$(RM) -f libs/libhypothesis_swift_core-maccatalyst.a
	@lipo -create -output libs/libhypothesis_swift_core-macos.a \
		target/aarch64-apple-darwin/release/libhypothesis_swift_core.a \
		target/x86_64-apple-darwin/release/libhypothesis_swift_core.a
	@lipo -create -output libs/libhypothesis_swift_core-maccatalyst.a \
		target/aarch64-apple-ios-macabi/release/libhypothesis_swift_core.a \
		target/x86_64-apple-ios-macabi/release/libhypothesis_swift_core.a

ios: common
	@rustup target add aarch64-apple-ios
	@rustup target add aarch64-apple-ios-sim
	@rustup target add x86_64-apple-ios
	@cargo build --release --lib --target aarch64-apple-ios
	@cargo build --release --lib --target aarch64-apple-ios-sim
	@cargo build --release --lib --target x86_64-apple-ios
	@$(RM) -f libs/libhypothesis_swift_core-ios.a
	@$(RM) -f libs/libhypothesis_swift_core-ios-sim.a
	@cp target/aarch64-apple-ios/release/libhypothesis_swift_core.a libs/libhypothesis_swift_core-ios.a
	@lipo -create -output libs/libhypothesis_swift_core-ios-sim.a \
		target/aarch64-apple-ios-sim/release/libhypothesis_swift_core.a \
		target/x86_64-apple-ios/release/libhypothesis_swift_core.a

tvos: common
	@cargo build build-std --release --lib --target aarch64-apple-tvos
	@cargo build build-std --release --lib --target aarch64-apple-tvos-sim
	@cargo build build-std --release --lib --target x86_64-apple-tvos
	@cargo build --release --lib --target aarch64-apple-tvos-sim
	@cargo build --release --lib --target x86_64-apple-tvos
	@$(RM) -f libs/libhypothesis_swift_core-tvos.a
	@$(RM) -f libs/libhypothesis_swift_core-tvos-sim.a
	@cp target/aarch64-apple-tvos/release/libhypothesis_swift_core.a libs/libhypothesis_swift_core-tvos.a
	@lipo -create -output libs/libhypothesis_swift_core-tvos-sim.a \
		target/aarch64-apple-tvos-sim/release/libhypothesis_swift_core.a \
		target/x86_64-apple-tvos/release/libhypothesis_swift_core.a

watchos: common
	@rustup target add aarch64-apple-watchos
	@rustup target add aarch64-apple-watchos-sim
	@rustup target add x86_64-apple-watchos
	@cargo build --release --lib --target aarch64-apple-watchos
	@cargo build --release --lib --target aarch64-apple-watchos-sim
	@cargo build --release --lib --target x86_64-apple-watchos
	@$(RM) -f libs/libhypothesis_swift_core-watchos.a
	@$(RM) -f libs/libhypothesis_swift_core-watchos-sim.a
	@cp target/aarch64-apple-watchos/release/libhypothesis_swift_core.a libs/libhypothesis_swift_core-watchos.a
	@lipo -create -output libs/libhypothesis_swift_core-watchos-sim.a \
		target/aarch64-apple-watchos-sim/release/libhypothesis_swift_core.a \
		target/x86_64-apple-watchos/release/libhypothesis_swift_core.a

visionos: common
	@rustup target add aarch64-apple-visionos
	@rustup target add aarch64-apple-visionos-sim
	@cargo build --release --lib --target aarch64-apple-visionos
	@cargo build --release --lib --target aarch64-apple-visionos-sim
	@$(RM) -f libs/libhypothesis_swift_core-visionos.a
	@$(RM) -f libs/libhypothesis_swift_core-visionos-sim.a
	@cp target/aarch64-apple-visionos/release/libhypothesis_swift_core.a libs/libhypothesis_swift_core-visionos.a
	@cp target/aarch64-apple-visionos-sim/release/libhypothesis_swift_core.a libs/libhypothesis_swift_core-visionos-sim.a
