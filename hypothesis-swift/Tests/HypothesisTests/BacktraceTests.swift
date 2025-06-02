import Testing
@testable import Hypothesis

@Suite
struct BacktraceTests {
    @Test
    func test() throws {
        try hypothesis {
            let int = try any(ConjectureIntegers.unbounded())
            print(int)
        }
    }
    
    @Test
    func throwing() {
        #expect(10 < 20)
    }
}
