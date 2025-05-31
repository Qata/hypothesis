import Foundation

public protocol HypothesisDistribution {
    associatedtype Value
    func provide(from source: ConjectureDataSource) throws -> Value
}

// MARK: - Integer Distributions

/// Distribution for unbounded integers
///
/// Design decisions:
/// 1. Making it a struct since it just wraps the underlying Conjecture type
/// 2. Conforming to HypothesisDistribution for use with generic draw method
/// 3. Using a static factory method instead of direct initialization
public struct Integers: HypothesisDistribution {
    private let conjectureIntegers: ConjectureIntegers
    
    /// Private initializer to ensure proper setup
    private init(_ integers: ConjectureIntegers) {
        self.conjectureIntegers = integers
    }
    
    /// Create a new unbounded integer distribution
    ///
    /// Design: Static factory method provides cleaner API than try init
    public static func unbounded() throws -> Integers {
        Integers(try ConjectureIntegers())
    }
    
    /// Provide a value from this distribution
    public func provide(from source: ConjectureDataSource) throws -> Int64 {
        try conjectureIntegers.provide(from: source)
    }
}

/// Distribution for bounded integers
///
/// Design decisions:
/// 1. Storing the bounds for reference (useful for debugging/testing)
/// 2. Supporting both single bound and range initialization
public struct BoundedIntegers: HypothesisDistribution {
    public let minValue: UInt64
    public let maxValue: UInt64
    private let conjectureBounded: ConjectureBoundedIntegers
    
    /// Initialize with a maximum value (minimum is 0)
    public init(maxValue: UInt64) throws {
        self.minValue = 0
        self.maxValue = maxValue
        self.conjectureBounded = try ConjectureBoundedIntegers(maxValue: maxValue)
    }
    
    /// Initialize with a range
    ///
    /// Design: This is more Swift-like than separate min/max parameters
    public init(range: ClosedRange<UInt64>) throws {
        guard range.lowerBound <= range.upperBound else {
            throw HypothesisError.invalidConfiguration("Range lower bound must be <= upper bound")
        }
        
        // Conjecture only supports 0-based bounded integers
        // So we'll need to offset the result
        self.minValue = range.lowerBound
        self.maxValue = range.upperBound
        let adjustedMax = range.upperBound - range.lowerBound
        self.conjectureBounded = try ConjectureBoundedIntegers(maxValue: adjustedMax)
    }
    
    /// Provide a value from this distribution
    public func provide(from source: ConjectureDataSource) throws -> UInt64 {
        let value = try conjectureBounded.provide(from: source)
        return value + minValue
    }
}

// MARK: - Collection Distributions

/// Configuration for generating collections
///
/// Design: Using a builder pattern for flexible configuration
public struct CollectionParameters {
    public let minSize: UInt64
    public let maxSize: UInt64
    public let expectedSize: Double
    
    public init(
        minSize: UInt64 = 0,
        maxSize: UInt64 = 10,
        expectedSize: Double? = nil
    ) {
        self.minSize = minSize
        self.maxSize = maxSize
        self.expectedSize = expectedSize ?? Double(minSize + maxSize) / 2.0
    }
    
    /// Create parameters for exact size
    public static func exactly(_ size: UInt64) -> CollectionParameters {
        CollectionParameters(minSize: size, maxSize: size, expectedSize: Double(size))
    }
    
    /// Create parameters for a range
    public static func range(_ range: ClosedRange<UInt64>) -> CollectionParameters {
        CollectionParameters(
            minSize: range.lowerBound,
            maxSize: range.upperBound
        )
    }
}

/// Distribution for arrays of values
///
/// Design decisions:
/// 1. Generic over the element distribution type
/// 2. Using a result builder pattern for configuration
public struct Arrays<Element>: HypothesisDistribution where Element: HypothesisDistribution {
    public typealias Value = [Element.Value]
    
    private let elementDistribution: Element
    private let parameters: CollectionParameters
    
    public init(
        of element: Element,
        parameters: CollectionParameters = CollectionParameters()
    ) {
        self.elementDistribution = element
        self.parameters = parameters
    }
    
    /// Convenience initializer with size range
    public init(
        of element: Element,
        size: ClosedRange<UInt64>
    ) {
        self.init(of: element, parameters: .range(size))
    }
    
    /// Convenience initializer for exact size
    public init(
        of element: Element,
        count: UInt64
    ) {
        self.init(of: element, parameters: .exactly(count))
    }
    
    public func provide(from source: ConjectureDataSource) throws -> [Element.Value] {
        let repeatValues = try ConjectureRepeatValues(
            minCount: parameters.minSize,
            maxCount: parameters.maxSize,
            expectedCount: parameters.expectedSize
        )
        
        var result: [Element.Value] = []
        
        while try repeatValues.shouldContinue(with: source) {
            try source.startDraw()
            let value = try elementDistribution.provide(from: source)
            try source.stopDraw()
            result.append(value)
        }
        
        return result
    }
}

// MARK: - Convenience Extensions

/// Extension to make creating distributions more fluent
///
/// Design: Adding static factory methods for common cases
public extension Integers {
    /// Create integers in a specific range
    ///
    /// Note: This creates a specialized distribution, not using BoundedIntegers
    static func inRange(_ range: ClosedRange<Int64>) throws -> AnyDistribution<Int64> {
        let base = try Integers.unbounded()
        
        return AnyDistribution { source in
            let value = try base.provide(from: source)
            
            // Simple modulo approach for now
            // A real implementation would use a more sophisticated approach
            let rangeSize = range.upperBound - range.lowerBound + 1
            let adjusted = abs(value) % rangeSize
            return range.lowerBound + adjusted
        }
    }
}

// MARK: - Type-Erased Distribution

/// Type-erased distribution for when you need to store different distributions
///
/// Design: Similar to AnyPublisher in Combine
public struct AnyDistribution<Value>: HypothesisDistribution {
    private let _provide: (ConjectureDataSource) throws -> Value
    
    public init<D: HypothesisDistribution>(_ distribution: D) where D.Value == Value {
        self._provide = distribution.provide
    }
    
    public init(provide: @escaping (ConjectureDataSource) throws -> Value) {
        self._provide = provide
    }
    
    public func provide(from source: ConjectureDataSource) throws -> Value {
        try _provide(source)
    }
}

// MARK: - DSL Support

/// Result builder for composing distributions
///
/// Design: Leveraging Swift's result builders for a DSL-like API
@resultBuilder
public struct DistributionBuilder {
    public static func buildBlock<D: HypothesisDistribution>(_ distribution: D) -> D {
        distribution
    }
}

// MARK: - Common Distributions

/// Namespace for common distribution factories
///
/// Design: Providing a convenient API similar to SwiftUI's views
public enum Distributions {
    /// Integers without bounds
    public static var integers: Integers {
        get throws {
            try Integers.unbounded()
        }
    }
    
    /// Integers from 0 to max
    public static func integers(upTo max: UInt64) throws -> BoundedIntegers {
        try BoundedIntegers(maxValue: max)
    }
    
    /// Integers in a range
    public static func integers(in range: ClosedRange<UInt64>) throws -> BoundedIntegers {
        try BoundedIntegers(range: range)
    }
    
    /// Arrays with default parameters
    public static func arrays<D: HypothesisDistribution>(
        of element: D
    ) -> Arrays<D> {
        Arrays(of: element)
    }
    
    /// Arrays with size range
    public static func arrays<D: HypothesisDistribution>(
        of element: D,
        size: ClosedRange<UInt64>
    ) -> Arrays<D> {
        Arrays(of: element, size: size)
    }
}
