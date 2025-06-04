public protocol Possible<Value> {
    associatedtype Value
    func provide(from source: CoreDataSource) throws -> Value
}

// MARK: - Possible Extensions

public extension Possible {
    /// A Possible value constructed by passing one of these Possible values to the provided block.
    ///
    /// Example: `integers.map { $0 * 2 }` creates a distribution of even integers.
    func map<T>(
        _ transform: @escaping (Value) throws -> T
    ) -> some Possible<T> {
        AnyPossible { source in
            try transform(
                provide(from: source)
            )
        }
    }
    
    func flatMap<T>(
        _ transform: @escaping (Value) throws -> any Possible<T>
    ) -> some Possible<T> {
        AnyPossible { source in
            try transform(
                provide(
                    from: source
                )
            ).provide(
                from: source
            )
        }
    }
    
    /// One of these Possible values selected such that the condition returns true for it.
    ///
    /// Example: `integers.filter { $0 % 2 == 0 }` creates a distribution of even integers.
    ///
    /// - Note: Similar warnings to `assume()` apply here: If the condition is difficult to satisfy
    ///   this may impact the performance and quality of your testing.
    func filter(
        _ condition: @escaping (Value) throws -> Bool
    ) -> some Possible<Value> {
        AnyPossible { source in
            let maxAttempts = 3
            for _ in 1...maxAttempts {
                let candidate = try self.provide(from: source)
                if try condition(candidate) {
                    return candidate
                }
            }
            throw HypothesisError.unsatisfiedAssumption
        }
    }
    
    func optional() -> AnyPossible<Value?> {
        Possibilities
            .optional(of: self)
            .erased()
    }
    
    func erased() -> AnyPossible<Value> {
        .init(self)
    }
}

// MARK: - Composite Possible Implementation

/// A Possible implementation that composes other Possibles or uses custom generation logic.
public struct AnyPossible<Value>: Possible {
    private let generator: (CoreDataSource) throws -> Value
    
    public init<P: Possible>(_ possible: P) where P.Value == Value {
        self.generator = possible.provide
    }
    
    public init(_ generator: @escaping (CoreDataSource) throws -> Value) {
        self.generator = generator
    }
    
    public func provide(from source: CoreDataSource) throws -> Value {
        try generator(source)
    }
}


public
extension Possible where Value: Possible {
    func flatten() -> AnyPossible<Value.Value> {
        AnyPossible {
            try provide(
                from: $0
            ).provide(
                from: $0
            )
        }
    }
}
