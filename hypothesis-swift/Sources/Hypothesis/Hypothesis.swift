import Conjecture

public enum Phase: CaseIterable {
    case shrink
    
    var cPhase: CPhase {
        switch self {
        case .shrink: .shrink
        }
    }
}

public func hypothesis(
    maxValidTestCases: UInt64 = 200,
    phases: [Phase] = Phase.allCases,
    databasePath: String? = nil,
    seed: UInt64 = UInt64.random(in: 0...UInt64.max),
    block: (TestCase) throws -> Void
) throws {
    guard World.currentEngine == nil else {
        throw HypothesisError.usageError("Cannot nest hypothesis calls")
    }
    defer { World.currentEngine = nil }
    World.currentEngine = try Engine(
        name: "",
        databasePath: databasePath,
        seed: seed,
        maxExamples: maxValidTestCases,
        phases: phases
    )
    try World.currentEngine?.run(block)
}

public func hypothesis(
    maxValidTestCases: UInt64 = 200,
    phases: [Phase] = Phase.allCases,
    databasePath: String? = nil,
    seed: UInt64 = UInt64.random(in: 0...UInt64.max),
    block: () throws -> Void
) throws {
    try hypothesis(
        maxValidTestCases: maxValidTestCases,
        phases: phases,
        databasePath: databasePath,
        seed: seed,
        block: { _ in try block() }
    )
}
