import Testing
@testable import Hypothesis

@Suite
struct BacktraceTests {
    @Test
    func test1() throws {
        try hypothesis {
            let int = try any(ConjectureIntegers.unbounded())
            print(int)
            try verify(int < 51)
        }
    }
    
    @Test
    func test2() throws {
        try hypothesis {
            let int = try any(ConjectureIntegers.unbounded())
            print(int)
        }
    }
}
