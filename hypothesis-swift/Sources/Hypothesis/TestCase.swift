public class TestCase {
    
    // MARK: - Ruby API Port: Properties
    
    /// The underlying C data source (Ruby: @wrapped_data)
    let wrappedData: CoreDataSource
    
    /// Generated values for replay (Ruby: @draws)
    private(set) var draws: [Any]?
    
    /// Pretty-printed log entries (Ruby: @print_log)
    private(set) var printLog: [(name: String?, value: String)]?
    
    /// Current nesting depth (Ruby: @depth)
    private var depth: Int = 0
    
    // MARK: - Initialization
    
    /// Initialize a new TestCase
    /// - Parameters:
    ///   - wrappedData: The underlying ConjectureDataSource
    ///   - printDraws: Whether to enable pretty-printing for error messages
    ///   - recordDraws: Whether to record draws for failure reproduction
    init(
        _ wrappedData: CoreDataSource,
        printDraws: Bool = false,
        recordDraws: Bool = false
    ) {
        self.wrappedData = wrappedData
        self.draws = recordDraws ? [] : nil
        self.printLog = printDraws ? [] : nil
        self.depth = 0
    }
    
    // MARK: - Ruby API Port: Core Methods
    
    /// Make an assumption about the current test case.
    /// Ruby equivalent: `def assume(condition)`
    /// - Parameter condition: The condition that must be true
    /// - Throws: `HypothesisError.unsatisfiedAssumption` if condition is false
    func assume(_ condition: Bool) throws {
        guard condition else {
            throw HypothesisError.unsatisfiedAssumption
        }
    }
    
    /// Generate a value using the provided distribution.
    /// Ruby equivalent: `def any(possible = nil, name: nil, &block)`
    /// - Parameters:
    ///   - distribution: The distribution to sample from
    ///   - name: Optional name for debugging/pretty-printing
    /// - Returns: A generated value
    /// - Throws: Various errors from the generation process
    func any<T: Possible>(
        _ distribution: T,
        name: String? = nil
    ) throws -> T.Value {
        let isTopLevel = depth == 0
        depth += 1
        defer { depth -= 1 }
        
        let result: T.Value
        try wrappedData.startDraw()
        do {
            result = try distribution.provide(from: wrappedData)
        } catch {
            throw error
        }
        try wrappedData.stopDraw()
        
        // Only log top-level draws (Ruby: if top_level)
        if isTopLevel {
            // Record for replay (Ruby: draws&.push(result))
            draws?.append(result)
            
            // Record for pretty-printing (Ruby: print_log&.push([name, result.inspect]))
            let displayName = name ?? "#\((printLog?.count ?? 0) + 1)"
            let valueDescription = String(describing: result)
            printLog?.append((name: displayName, value: valueDescription))
        }
        
        return result
    }
}
