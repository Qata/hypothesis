import Conjecture
import Foundation

public class Engine {
    private(set) var currentTestCase: TestCase?
    private let conjectureEngine: ConjectureEngine
    private let name: String
    private var isFind: Bool = false
    private var exceptionToTags: AutoIncrementingDictionary<Backtrace> = .init()

    public init(
        name: String,
        databasePath: String?,
        seed: UInt64,
        maxExamples: UInt64,
        phases: [Phase]
    ) throws {
        self.name = name
        self.conjectureEngine = try ConjectureEngine(
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
            } catch HypothesisError.dataOverflow {
                try conjectureEngine.finish(source, .overflow)
            } catch {
                let backtraced = BacktracedError(underlying: error)
                if isFind {
                    throw backtraced
                }
                try conjectureEngine.finish(
                    source, .interesting(
                        label: exceptionToTags[backtraced.backtrace]
                    )
                )
            }
        }
        
        if try conjectureEngine.countFailingExamples() == 0 {
            if try conjectureEngine.wasUnsatisfiable() {
                throw HypothesisError.unsatisfiable
            }
            self.currentTestCase = nil
            return
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
                    _ = try testBlock(currentTestCase!)
                } catch {
                    let givens = currentTestCase?.printLog ?? []
                    let givenStrings = givens.enumerated().map { (index, item) in
                        let name = item.name ?? "#\(index + 1)"
                        return "Given \(name): \(item.value)"
                    }
                    
                    let enhancedError = error as? HypothesisEnhancedError
                    ?? HypothesisWrappedError(
                        originalError: error,
                        givens: givenStrings,
                        originalDescription: String(describing: error),
                        originalDebugDescription: String(reflecting: error)
                    )
                    if try conjectureEngine.countFailingExamples() == 1 {
                        throw enhancedError
                    }
                    exceptions.append(enhancedError)
                }
            }
            throw HypothesisMultipleExceptionError(exceptions)
        }
    }
}
