public class HypothesisDataSource {
    private var wrappedSource: ConjectureDataSource?
    
    /// Access the underlying source, throwing if it has been consumed
    private var source: ConjectureDataSource {
        get throws {
            guard let source = wrappedSource else {
                throw HypothesisError.sourceConsumed
            }
            return source
        }
    }
    
    /// Initialize with an optional source from the engine
    init(_ source: ConjectureDataSource?) {
        self.wrappedSource = source
    }
    
    /// Start a new draw operation
    ///
    /// Design: Using throwing functions instead of Ruby's implicit exceptions
    public func startDraw() throws {
        try source.startDraw()
    }
    
    /// Stop the current draw operation
    public func stopDraw() throws {
        try source.stopDraw()
    }
    
    /// Draw a value from a distribution
    ///
    /// Design: Generic method that works with any HypothesisDistribution
    public func draw<Distribution>(
        _ distribution: Distribution
    ) throws -> Distribution.Value
    where Distribution: HypothesisDistribution {
        try distribution.provide(from: source)
    }
    
    /// Draw raw bits
    public func bits(_ count: UInt64) throws -> UInt64 {
        try source.bits(count)
    }
    
    /// Write a deterministic value
    public func write(_ value: UInt64) throws {
        try source.write(value)
    }
    
    /// Consume the source and return it
    ///
    /// Design: Using Swift's consume semantics to make ownership transfer explicit
    internal func consume() -> ConjectureDataSource? {
        defer { wrappedSource = nil }
        return wrappedSource
    }
}
