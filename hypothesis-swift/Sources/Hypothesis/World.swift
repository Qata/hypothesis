import Foundation

enum World {
    private static let threadKey = "HypothesisCurrentEngine"
    
    static var currentEngine: Engine? {
        get {
            Thread.current.threadDictionary[threadKey] as? Engine
        }
        set {
            if let newValue = newValue {
                Thread.current.threadDictionary[threadKey] = newValue
            } else {
                Thread.current.threadDictionary.removeObject(forKey: threadKey)
            }
        }
    }
    
    static func requireEngine() throws -> Engine {
        guard let engine = currentEngine else {
            throw HypothesisError.outsideTestContext
        }
        return engine
    }
}

// MARK: - Updated Global Functions

/// Generate a value using the current engine's test case
public func any<T: ConjectureDistribution>(
    _ distribution: T,
    _ name: String? = nil
) throws -> T.Value {
    let engine = try World.requireEngine()
    guard let testCase = engine.currentTestCase else {
        throw HypothesisError.internal
    }
    return try testCase.any(distribution, name: name)
}

/// Make an assumption about the current test case
public func assume(_ condition: Bool) throws {
    let engine = try World.requireEngine()
    guard let testCase = engine.currentTestCase else {
        throw HypothesisError.internal
    }
    try testCase.assume(condition)
}

public func verify(
    _ condition: Bool,
    _ message: String? = nil,
    file: String = #file,
    line: Int = #line,
    column: Int = #column
) throws {
    let engine = try World.requireEngine()
    guard engine.currentTestCase != nil else {
        throw HypothesisError.internal
    }
    guard condition else {
        throw HypothesisError.unverifiable(
            message,
            location: "\(file):\(line):\(column)"
        )
    }
}
