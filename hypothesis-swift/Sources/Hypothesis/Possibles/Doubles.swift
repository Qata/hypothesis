import math_h

public struct DoublesPossible: Possible {
    public typealias Value = Double
    
    private let minValue: Double
    private let maxValue: Double
    private let allowNaN: Bool
    private let allowInfinity: Bool
    private let allowSubnormal: Bool
    private let coreBounded: CoreBoundedIntegers
    private let coreUnbounded: CoreIntegers
    
    // IEEE 754 double precision constants
    private static let exponentBits: UInt64 = 11
    private static let mantissaBits: UInt64 = 52
    private static let signBit: UInt64 = 1 << 63
    private static let exponentMask: UInt64 = ((1 << exponentBits) - 1) << mantissaBits
    private static let mantissaMask: UInt64 = (1 << mantissaBits) - 1
    private static let smallestNormal: Double = 2.2250738585072014e-308
    private static let smallestSubnormal: Double = 4.9406564584124654e-324
    
    public init(
        min: Double? = nil,
        max: Double? = nil,
        allowNaN: Bool? = nil,
        allowInfinity: Bool? = nil,
        allowSubnormal: Bool? = nil
    ) throws {
        // Handle default values like Python implementation
        // Only allow NaN by default when both min/max are nil AND allowInfinity is not explicitly set to true
        let resolvedAllowNaN = allowNaN ?? (min == nil && max == nil && allowInfinity != true)
        let resolvedAllowInfinity = allowInfinity ?? (min == nil || max == nil)
        
        // Validate arguments
        if resolvedAllowNaN && (min != nil || max != nil) {
            throw HypothesisError.usageError("Cannot have allowNaN=true with min or max bounds")
        }
        
        if resolvedAllowInfinity && min != nil && max != nil {
            throw HypothesisError.usageError("Cannot have allowInfinity=true with both min and max bounds")
        }
        
        // Set bounds, using infinity as defaults
        let resolvedMin: Double
        let resolvedMax: Double
        
        if let min = min {
            guard min.isFinite || min.isInfinite else {
                throw HypothesisError.usageError("min must be finite or infinite, not NaN")
            }
            resolvedMin = min
        } else {
            resolvedMin = resolvedAllowInfinity ? -Double.infinity : -Double.greatestFiniteMagnitude
        }
        
        if let max = max {
            guard max.isFinite || max.isInfinite else {
                throw HypothesisError.usageError("max must be finite or infinite, not NaN")
            }
            resolvedMax = max
        } else {
            resolvedMax = resolvedAllowInfinity ? Double.infinity : Double.greatestFiniteMagnitude
        }
        
        guard resolvedMin <= resolvedMax else {
            throw HypothesisError.usageError("min must be <= max")
        }
        
        // Determine subnormal handling
        let resolvedAllowSubnormal: Bool
        if let allowSubnormal = allowSubnormal {
            resolvedAllowSubnormal = allowSubnormal
        } else {
            // Auto-detect based on range
            if resolvedMin.isFinite && resolvedMax.isFinite {
                if resolvedMin == resolvedMax {
                    resolvedAllowSubnormal = abs(resolvedMin) < Self.smallestNormal
                } else {
                    resolvedAllowSubnormal = resolvedMin < Self.smallestNormal && resolvedMax > -Self.smallestNormal
                }
            } else if resolvedMin.isFinite {
                resolvedAllowSubnormal = resolvedMin < Self.smallestNormal
            } else if resolvedMax.isFinite {
                resolvedAllowSubnormal = resolvedMax > -Self.smallestNormal
            } else {
                resolvedAllowSubnormal = true
            }
        }
        
        self.minValue = resolvedMin
        self.maxValue = resolvedMax
        self.allowNaN = resolvedAllowNaN
        self.allowInfinity = resolvedAllowInfinity
        self.allowSubnormal = resolvedAllowSubnormal
        
        // We need this for various random choices
        self.coreBounded = try CoreBoundedIntegers(maxValue: UInt64.max)
        self.coreUnbounded = try CoreIntegers()
    }
    
    public func provide(from source: CoreDataSource) throws -> Double {
        // Handle NaN generation (similar to Python's NanStrategy)
        if allowNaN {
            // Small chance of generating NaN
            let shouldGenerateNaN = try coreBounded.provide(from: source) % 100 == 0
            if shouldGenerateNaN {
                return try generateNaN(from: source)
            }
        }
        
        // Handle infinity generation
        if allowInfinity {
            let infinityChoice = try coreBounded.provide(from: source) % 1000
            if infinityChoice < 5 {  // Small chance for positive infinity
                let value = Double.infinity
                return (value >= minValue && value <= maxValue) ? value : try generateFiniteValue(from: source)
            } else if infinityChoice < 10 {  // Small chance for negative infinity
                let value = -Double.infinity
                return (value >= minValue && value <= maxValue) ? value : try generateFiniteValue(from: source)
            }
        }
        
        // Generate finite values
        return try generateFiniteValue(from: source)
    }
    
    private func generateNaN(from source: CoreDataSource) throws -> Double {
        // Generate NaN similar to Python's NanStrategy
        let signBit = (try coreBounded.provide(from: source) & 1) << 63
        let nanBits = Double.nan.bitPattern & ~Self.signBit  // Remove sign bit
        let mantissaBits = try coreBounded.provide(from: source) & Self.mantissaMask
        
        let result = Double(bitPattern: signBit | nanBits | mantissaBits)
        return result.isNaN ? result : Double.nan
    }
    
    private func generateFiniteValue(from source: CoreDataSource) throws -> Double {
        // Handle special case where min == max
        if minValue == maxValue && minValue.isFinite {
            return minValue
        }
        
        // Handle bounded finite ranges
        if minValue.isFinite && maxValue.isFinite {
            return try generateBoundedFiniteValue(from: source)
        }
        
        // Handle unbounded or semi-bounded ranges
        return try generateUnboundedValue(from: source)
    }
    
    private func generateBoundedFiniteValue(from source: CoreDataSource) throws -> Double {
        // Use linear interpolation for bounded ranges (like original implementation)
        let randomBits = try coreBounded.provide(from: source)
        let fraction = Double(randomBits) / Double(UInt64.max)
        
        let result = minValue + fraction * (maxValue - minValue)
        return Swift.max(minValue, Swift.min(maxValue, result))
    }
    
    private func generateUnboundedValue(from source: CoreDataSource) throws -> Double {
        // Strategy similar to Python's draw_float implementation
        // This is a simplified version - the real Python implementation is quite complex
        
        // Generate the components of a float
        let signChoice = try coreBounded.provide(from: source) % 2
        let sign: Double
        
        // Respect bounds when choosing sign
        if minValue >= 0.0 {
            // Only positive values allowed
            sign = 1.0
        } else if maxValue <= 0.0 {
            // Only negative values allowed
            sign = -1.0
        } else {
            // Both positive and negative allowed
            sign = signChoice == 0 ? 1.0 : -1.0
        }
        
        // Choose between different magnitude ranges
        let magnitudeChoice = try coreBounded.provide(from: source) % 1000
        
        let magnitude: Double
        if magnitudeChoice < 100 {
            // Small values (including subnormals if allowed)
            let smallValue = try generateSmallMagnitude(from: source)
            magnitude = smallValue
        } else if magnitudeChoice < 200 {
            // Medium values around 1.0
            let mediumValue = try generateMediumMagnitude(from: source)
            magnitude = mediumValue
        } else if magnitudeChoice < 300 {
            // Large values
            let largeValue = try generateLargeMagnitude(from: source)
            magnitude = largeValue
        } else {
            // Integer-like values for nice shrinking
            let intValue = Int64(try coreBounded.provide(from: source) % 1000)
            magnitude = Double(intValue)
        }
        
        let result = sign * magnitude
        
        // Ensure result is within bounds
        if result < minValue || result > maxValue {
            // Fallback to bounded generation if we're out of range
            if minValue.isFinite && maxValue.isFinite {
                return try generateBoundedFiniteValue(from: source)
            } else {
                // Try again with a different approach
                return try generateSimpleUnboundedValue(from: source)
            }
        }
        
        return result
    }
    
    private func generateSmallMagnitude(from source: CoreDataSource) throws -> Double {
        if allowSubnormal {
            // Generate values that might be subnormal
            let exponent = -308 + Int(try coreBounded.provide(from: source) % 100)
            return pow(10.0, Double(exponent))
        } else {
            // Stay above smallest normal
            let exponent = -307 + Int(try coreBounded.provide(from: source) % 50)
            return pow(10.0, Double(exponent))
        }
    }
    
    private func generateMediumMagnitude(from source: CoreDataSource) throws -> Double {
        // Values around 0.1 to 100
        let exponent = -1 + Int(try coreBounded.provide(from: source) % 4)
        let mantissa = 1.0 + Double(try coreBounded.provide(from: source) % 1000) / 1000.0
        return mantissa * pow(10.0, Double(exponent))
    }
    
    private func generateLargeMagnitude(from source: CoreDataSource) throws -> Double {
        // Large finite values
        let exponent = 2 + Int(try coreBounded.provide(from: source) % 100)
        let mantissa = 1.0 + Double(try coreBounded.provide(from: source) % 1000) / 1000.0
        return mantissa * pow(10.0, Double(exponent))
    }
    
    private func generateSimpleUnboundedValue(from source: CoreDataSource) throws -> Double {
        // Fallback: simple conversion from integer
        let intValue = try coreUnbounded.provide(from: source)
        let scaleFactor = pow(10.0, Double(Int(try coreBounded.provide(from: source) % 21) - 10))
        let result = Double(intValue) * scaleFactor
        
        // Ensure result respects bounds
        if result < minValue || result > maxValue {
            // If we still can't generate a valid value, use bounded generation
            if minValue.isFinite && maxValue.isFinite {
                return try generateBoundedFiniteValue(from: source)
            } else {
                // Clamp to bounds
                return Swift.max(minValue, Swift.min(maxValue, result))
            }
        }
        
        return result
    }
}

// Add to Possibilities extension
public extension Possibilities {
    static func doubles(
        min: Double? = nil,
        max: Double? = nil,
        allowNaN: Bool? = nil,
        allowInfinity: Bool? = nil,
        allowSubnormal: Bool? = nil
    ) -> some Possible<Double> {
        AnyPossible { source in
            try DoublesPossible(
                min: min,
                max: max,
                allowNaN: allowNaN,
                allowInfinity: allowInfinity,
                allowSubnormal: allowSubnormal
            ).provide(from: source)
        }
    }
    
    // Convenience methods for common ranges
    static func doubles(
        in range: ClosedRange<Double>
    ) -> some Possible<Double> {
        doubles(min: range.lowerBound, max: range.upperBound, allowNaN: false, allowInfinity: false)
    }
    
    // For positive doubles only
    static func positiveDoubles(
        max: Double? = nil,
        allowInfinity: Bool? = nil
    ) -> some Possible<Double> {
        doubles(min: 0.0, max: max, allowNaN: false, allowInfinity: allowInfinity)
    }
    
    // For unit interval [0, 1]
    static func unitDoubles() -> some Possible<Double> {
        doubles(min: 0.0, max: 1.0, allowNaN: false, allowInfinity: false)
    }
    
    // For normalized range [-1, 1]
    static func normalizedDoubles() -> some Possible<Double> {
        doubles(min: -1.0, max: 1.0, allowNaN: false, allowInfinity: false)
    }
    
    // For finite doubles only (no NaN or infinity)
    static func finiteDoubles(
        min: Double? = nil,
        max: Double? = nil
    ) -> some Possible<Double> {
        doubles(min: min, max: max, allowNaN: false, allowInfinity: false)
    }
}
