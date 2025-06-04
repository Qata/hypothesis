import Testing
@testable import Hypothesis

@Suite
struct BacktraceTests {
    @Test
    func test1() throws {
        try hypothesis {
            let int = try any(CoreIntegers.unbounded())
            let int2 = try any(CoreIntegers.unbounded())
            try verify(int < int2, "Just checking")
        }
    }
    
    @Test
    func test2() throws {
        try hypothesis {
            let int = try any(CoreIntegers.unbounded())
        }
    }
}
