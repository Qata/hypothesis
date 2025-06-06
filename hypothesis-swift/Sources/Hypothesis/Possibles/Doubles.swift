public struct DoublesPossible: Possible {
    public typealias Value = Double
    
    private let minValue: Double
    private let maxValue: Double
    private let allowNaN: Bool
    private let allowInfinity: Bool
    private let allowSubnormal: Bool
    private let coreFloats: CoreFloats
    
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
            // Auto-detect based on range: always allow for now since Rust handles this
            resolvedAllowSubnormal = true
        }
        
        self.minValue = resolvedMin
        self.maxValue = resolvedMax
        self.allowNaN = resolvedAllowNaN
        self.allowInfinity = resolvedAllowInfinity
        self.allowSubnormal = resolvedAllowSubnormal
        
        // Initialize Rust float generator
        self.coreFloats = try CoreFloats()
    }
    
    public func provide(from source: CoreDataSource) throws -> Double {
        // Use sophisticated Rust float generation with lexicographic encoding
        return try coreFloats.provideBounded(
            from: source, 
            min: minValue, 
            max: maxValue, 
            allowNaN: allowNaN, 
            allowInfinity: allowInfinity
        )
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
    
    // Advanced Rust-backed generation methods
    
    /// Generate any double using sophisticated lexicographic encoding from Rust core
    /// This provides excellent shrinking properties but may generate any valid double
    static func anyDouble() -> some Possible<Double> {
        AnyPossible { source in
            try CoreFloats().provideAny(from: source)
        }
    }
    
    /// Generate doubles uniformly distributed in the given range
    /// Uses Rust's uniform generation which may have different properties than bounded generation
    static func uniformDoubles(min: Double, max: Double) -> some Possible<Double> {
        AnyPossible { source in
            try CoreFloats().provideUniform(from: source, min: min, max: max)
        }
    }
    
    /// Generate doubles that may include subnormal values
    /// Uses Rust core which properly handles subnormal numbers according to IEEE 754
    static func subnormalDoubles(
        min: Double? = nil,
        max: Double? = nil
    ) -> some Possible<Double> {
        doubles(min: min, max: max, allowNaN: false, allowInfinity: false, allowSubnormal: true)
    }
    
    /// Generate doubles including special IEEE 754 values (NaN, ±∞, ±0, subnormals)
    /// This uses the full power of the Rust lexicographic encoding
    static func specialDoubles() -> some Possible<Double> {
        doubles(allowNaN: true, allowInfinity: true, allowSubnormal: true)
    }
}
