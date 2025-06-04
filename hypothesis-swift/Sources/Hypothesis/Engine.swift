import Conjecture
import Foundation

public class Engine {
    private(set) var currentTestCase: TestCase?
    private let conjectureEngine: CoreEngine
    private let name: String
    private let maxExamples: UInt64
    private var isFind: Bool = false
    private var exceptionToTags: AutoIncrementingDictionary<String> = .init()

    public init(
        name: String,
        databasePath: String?,
        seed: UInt64,
        maxExamples: UInt64,
        phases: [Phase]
    ) throws {
        self.name = name
        self.maxExamples = maxExamples
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
                try conjectureEngine.finish(
                    source,
                    .interesting(
                        label: exceptionToTags[
                            location + (message.map { ":\($0)" } ?? "")
                        ]
                    )
                )
            } catch {
                try conjectureEngine.finish(
                    source,
                    .interesting(
                        label: exceptionToTags[
                            String(reflecting: error) + String(describing: error)
                        ]
                    )
                )
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
            
            for example in try 0..<conjectureEngine.countFailingExamples() {
                let source = try conjectureEngine.failingExample(atIndex: example)
                self.currentTestCase = TestCase(source, printDraws: true)
                do {
                    try currentTestCase.map(testBlock)
                } catch {
                    let wrappedError = HypothesisWrappedError(
                        error,
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
