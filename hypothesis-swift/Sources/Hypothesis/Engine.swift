import Conjecture
import Foundation

public class Engine {
    private(set) var currentTestCase: TestCase?
    private let conjectureEngine: CoreEngine
    private let name: String
    private let maxExamples: UInt64
    private let maxFailingExamples: UInt64
    private var isFind: Bool = false
    private var exceptionToTags: AutoIncrementingDictionary<String> = .init()
    private var interestingExampleCount: UInt64 = 0

    public init(
        name: String,
        databasePath: String?,
        seed: UInt64,
        maxExamples: UInt64,
        maxFailingExamples: UInt64 = 500,
        phases: [Phase]
    ) throws {
        self.name = name
        self.maxExamples = maxExamples
        self.maxFailingExamples = maxFailingExamples
        self.conjectureEngine = try CoreEngine(
            name: name,
            databasePath: databasePath,
            seed: seed,
            maxExamples: maxExamples,
            phases: phases.map(\.cPhase)
        )
    }
    
    func run(_ testBlock: (TestCase) throws -> Void) throws {
        while let source = try conjectureEngine.newSource() {
            self.currentTestCase = TestCase(source)
            do {
                try testBlock(currentTestCase!)
                if isFind {
                    try conjectureEngine.finish(source, .interesting(label: 0))
                } else {
                    try conjectureEngine.finish(source, .valid)
                }
            } catch HypothesisError.unsatisfiedAssumption {
                try conjectureEngine.finish(source, .invalid)
            } catch CoreError.dataOverflow {
                try conjectureEngine.finish(source, .overflow)
            } catch let HypothesisError.unverifiable(message, location) {
                if isFind {
                    throw HypothesisError.unverifiable(message, location: location)
                }
                
                if interestingExampleCount < maxFailingExamples {
                    interestingExampleCount += 1
                    try conjectureEngine.finish(
                        source,
                        .interesting(
                            label: exceptionToTags[
                                location + (message.map { ":\($0)" } ?? "")
                            ]
                        )
                    )
                } else {
                    // Mark as invalid to prevent collection of more examples
                    try conjectureEngine.finish(source, .invalid)
                }
            } catch {
                if interestingExampleCount < maxFailingExamples {
                    interestingExampleCount += 1
                    try conjectureEngine.finish(
                        source,
                        .interesting(
                            label: exceptionToTags[
                                String(reflecting: error) + String(describing: error)
                            ]
                        )
                    )
                } else {
                    // Mark as invalid to prevent collection of more examples
                    try conjectureEngine.finish(source, .invalid)
                }
            }
        }
        
        guard try conjectureEngine.countFailingExamples() != 0 else {
            if try conjectureEngine.wasUnsatisfiable() {
                throw HypothesisError.unsatisfiable
            }
            self.currentTestCase = nil
            return print(
                "􁁛 Passed \(maxExamples) tests"
            )
        }
        
        if isFind {
            let source = try conjectureEngine.failingExample(atIndex: 0)
            self.currentTestCase = TestCase(source, recordDraws: true)
            try testBlock(currentTestCase!)
        } else {
            var exceptions: [Error] = []
            let failingExampleCount = try conjectureEngine.countFailingExamples()
            
            print(
            """
            +++Interesting examples: \(interestingExampleCount)
            +++Max failing examples: \(maxFailingExamples)
            +++Failing examples: \(failingExampleCount)
            """
            )
            
            if interestingExampleCount >= maxFailingExamples && failingExampleCount > 0 {
                print("⚠️ Failing example collection limited to \(maxFailingExamples) examples during test execution")
            }
            
            for example in try 0..<conjectureEngine.countFailingExamples() {
                let source = try conjectureEngine.failingExample(atIndex: example)
                self.currentTestCase = TestCase(source, printDraws: true)
                do {
                    try currentTestCase.map(testBlock)
                } catch let caughtError {
                    let wrappedError = HypothesisWrappedError(
                        caughtError,
                        givens: currentTestCase?
                            .printLog?
                            .enumerated()
                            .map { index, item in
                                let name = item.name ?? "#\(index + 1)"
                                return "Given \(name): \(item.value)"
                            }
                        ?? []
                    )
                    
                    if try conjectureEngine.countFailingExamples() == 1 {
                        throw wrappedError
                    }
                    exceptions.append(wrappedError)
                }
            }
            
            throw HypothesisMultipleExceptionError(exceptions)
        }
    }
}
