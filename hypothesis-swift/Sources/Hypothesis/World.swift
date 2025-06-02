class World {
    nonisolated(unsafe)
    static let shared: World = World()
    
    /// The currently active Engine, if any
    var currentEngine: Engine?
    
    private init() {}
    
    /// Sets the current engine for the duration of a test
    func setEngine(_ engine: Engine?) {
        currentEngine = engine
    }
    
    /// Gets the current engine, throwing if none is set
    func requireEngine() throws -> Engine {
        guard let engine = currentEngine else {
            throw HypothesisError.outsideTestContext
        }
        return engine
    }
    
    /// Safely executes a block with the given engine set as current
    func withEngine<T>(
        _ engine: Engine?,
        execute: () throws -> T
    ) rethrows -> T {
        let previousEngine = currentEngine
        setEngine(engine)
        defer { setEngine(previousEngine) }
        return try execute()
    }
}

// MARK: - Updated Global Functions

/// Generate a value using the current engine's test case
func any<T: ConjectureDistribution>(
    _ distribution: T,
    name: String? = nil
) throws -> T.Value {
    let engine = try World.shared.requireEngine()
    guard let testCase = engine.currentTestCase else {
        throw HypothesisError.internal
    }
    return try testCase.any(distribution, name: name)
}

/// Make an assumption about the current test case
func assume(_ condition: Bool) throws {
    let engine = try World.shared.requireEngine()
    guard let testCase = engine.currentTestCase else {
        throw HypothesisError.internal
    }
    try testCase.assume(condition)
}
