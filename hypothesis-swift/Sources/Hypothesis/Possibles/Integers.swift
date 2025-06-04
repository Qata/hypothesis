public struct IntegersPossible: Possible {
    public typealias Value = Int64
    
    private let coreIntegers: CoreIntegers
    
    public init() throws {
        self.coreIntegers = try CoreIntegers()
    }
    
    public func provide(from source: CoreDataSource) throws -> Int64 {
        try coreIntegers.provide(from: source)
    }
}

public struct BoundedIntegersPossible: Possible {
    public typealias Value = UInt64
    
    private let coreBounded: CoreBoundedIntegers
    
    public init(maxValue: UInt64) throws {
        self.coreBounded = try CoreBoundedIntegers(maxValue: maxValue)
    }
    
    public init(range: ClosedRange<UInt64>) throws {
        self.coreBounded = try CoreBoundedIntegers(range: range)
    }
    
    public func provide(from source: CoreDataSource) throws -> UInt64 {
        try coreBounded.provide(from: source)
    }
}
