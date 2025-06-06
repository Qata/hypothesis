import Testing
@testable import Hypothesis

struct TestStruct {
    let string: String
    let bool: Bool
    let dict: [String: [Bool]]?
}

@Suite
struct HypothesisTests {
    @Test
    func test1() throws {
        try hypothesis {
            _ = try any(
                Possibilities.tuples(
                    Possibilities.strings(),
                    Possibilities.bools(),
                    Possibilities.dictionaries(
                        ofShape: [
                            "key": Possibilities.arrays(
                                of: Possibilities.bools()
                            )
                        ]
                    ).optional()
                )
                
            )
            let int = try any(Possibilities.integers(min: .min, max: .max))
            let int2 = try any(Possibilities.integers())
            try any(Possibilities.doubles())
            try assume(int > int2)
            try verify(int > int2, "Just checking")
        }
    }
    
    @Test
    func test2() throws {
        try hypothesis {
            let int = try any(CoreIntegers())
        }
    }
}
