struct HypothesisMultipleExceptionError: Error, CustomStringConvertible {
    let errors: [Error]
    
    init(_ errors: [Error]) {
        self.errors = errors
    }
    
    var description: String {
        """
        Test raised \(errors.count) distinct errors:
        
        \(errors.map(String.init(describing:)).joined(separator: "\n"))
        """
    }
}
