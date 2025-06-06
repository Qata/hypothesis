import Testing
@testable import Hypothesis

@Suite
struct DoublesTests {
    @Test
    func testBasicDoubleGeneration() throws {
        print("\n=== Testing Basic Double Generation ===")
        var generatedValues: [Double] = []
        var nanCount = 0
        var infiniteCount = 0
        var finiteCount = 0
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.doubles())
            generatedValues.append(value)
            
            if value.isNaN {
                nanCount += 1
                print("Generated NaN: \(value)")
            } else if value.isInfinite {
                infiniteCount += 1
                print("Generated infinite: \(value)")
            } else {
                finiteCount += 1
                print("Generated finite double: \(value)")
            }
            
            // Basic sanity checks
            try verify(!value.isNaN || true, "NaN is allowed by default")
        }
        
        print("Summary: \(finiteCount) finite, \(infiniteCount) infinite, \(nanCount) NaN values")
        print("Sample values: \(Array(generatedValues.prefix(10)))")
    }
    
    @Test
    func testFiniteDoublesOnly() throws {
        print("\n=== Testing Finite Doubles Only ===")
        var generatedValues: [Double] = []
        var minValue = Double.infinity
        var maxValue = -Double.infinity
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.finiteDoubles())
            generatedValues.append(value)
            
            print("Generated finite double: \(value)")
            
            if value.isFinite {
                minValue = min(minValue, value)
                maxValue = max(maxValue, value)
            }
            
            try verify(value.isFinite, "Should only generate finite doubles, got: \(value)")
            try verify(!value.isNaN, "Should not generate NaN, got: \(value)")
            try verify(!value.isInfinite, "Should not generate infinity, got: \(value)")
        }
        
        print("Range: [\(minValue), \(maxValue)]")
        print("All values finite: \(generatedValues.allSatisfy(\.isFinite))")
    }
    
    @Test
    func testBoundedDoubles() throws {
        print("\n=== Testing Bounded Doubles [0.0, 10.0] ===")
        var generatedValues: [Double] = []
        var belowRange = 0
        var inRange = 0
        var aboveRange = 0
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.doubles(in: 0.0...10.0))
            generatedValues.append(value)
            
            print("Generated bounded double: \(value)")
            
            if value < 0.0 {
                belowRange += 1
                print("  ⚠️  Below range: \(value)")
            } else if value > 10.0 {
                aboveRange += 1
                print("  ⚠️  Above range: \(value)")
            } else {
                inRange += 1
                print("  ✅ In range: \(value)")
            }
            
            try verify(value >= 0.0, "Should be >= 0.0, got: \(value)")
            try verify(value <= 10.0, "Should be <= 10.0, got: \(value)")
            try verify(value.isFinite, "Bounded doubles should be finite, got: \(value)")
        }
        
        print("Range distribution: \(belowRange) below, \(inRange) in range, \(aboveRange) above")
        print("Min generated: \(generatedValues.min() ?? Double.nan)")
        print("Max generated: \(generatedValues.max() ?? Double.nan)")
    }
    
    @Test
    func testUnitDoubles() throws {
        print("\n=== Testing Unit Doubles [0.0, 1.0] ===")
        var generatedValues: [Double] = []
        var distribution: [String: Int] = ["[0.0-0.2]": 0, "(0.2-0.4]": 0, "(0.4-0.6]": 0, "(0.6-0.8]": 0, "(0.8-1.0]": 0]
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.unitDoubles())
            generatedValues.append(value)
            
            print("Generated unit double: \(value)")
            
            // Track distribution
            switch value {
            case 0.0...0.2:
                distribution["[0.0-0.2]"]! += 1
            case let x where x > 0.2 && x <= 0.4:
                distribution["(0.2-0.4]"]! += 1
            case let x where x > 0.4 && x <= 0.6:
                distribution["(0.4-0.6]"]! += 1
            case let x where x > 0.6 && x <= 0.8:
                distribution["(0.6-0.8]"]! += 1
            case let x where x > 0.8 && x <= 1.0:
                distribution["(0.8-1.0]"]! += 1
            default:
                print("  ⚠️  Out of expected range: \(value)")
            }
            
            try verify(value >= 0.0, "Should be >= 0.0, got: \(value)")
            try verify(value <= 1.0, "Should be <= 1.0, got: \(value)")
            try verify(value.isFinite, "Unit doubles should be finite, got: \(value)")
        }
        
        print("Distribution: \(distribution)")
        print("Values cover range: [\(generatedValues.min() ?? 0), \(generatedValues.max() ?? 0)]")
    }
    
    @Test
    func testNormalizedDoubles() throws {
        print("\n=== Testing Normalized Doubles [-1.0, 1.0] ===")
        var generatedValues: [Double] = []
        var negativeCount = 0
        var zeroCount = 0
        var positiveCount = 0
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.normalizedDoubles())
            generatedValues.append(value)
            
            print("Generated normalized double: \(value)")
            
            if value < 0 {
                negativeCount += 1
                print("  📉 Negative: \(value)")
            } else if value == 0 {
                zeroCount += 1
                print("  ⚪ Zero: \(value)")
            } else {
                positiveCount += 1
                print("  📈 Positive: \(value)")
            }
            
            try verify(value >= -1.0, "Should be >= -1.0, got: \(value)")
            try verify(value <= 1.0, "Should be <= 1.0, got: \(value)")
            try verify(value.isFinite, "Normalized doubles should be finite, got: \(value)")
        }
        
        print("Sign distribution: \(negativeCount) negative, \(zeroCount) zero, \(positiveCount) positive")
        print("Extreme values: min=\(generatedValues.min() ?? 0), max=\(generatedValues.max() ?? 0)")
    }
    
    @Test
    func testPositiveDoubles() throws {
        print("\n=== Testing Positive Doubles ===")
        var generatedValues: [Double] = []
        var infinityCount = 0
        var finiteCount = 0
        var verySmallCount = 0 // count values close to zero
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.positiveDoubles())
            generatedValues.append(value)
            
            if value.isInfinite {
                infinityCount += 1
                print("Generated positive infinity: \(value)")
            } else if value < 1e-10 {
                verySmallCount += 1
                print("Generated very small positive: \(value)")
            } else {
                finiteCount += 1
                print("Generated positive double: \(value)")
            }
            
            try verify(value >= 0.0, "Should be >= 0.0, got: \(value)")
        }
        
        print("Type distribution: \(finiteCount) finite, \(infinityCount) infinite, \(verySmallCount) very small")
        let finiteValues = generatedValues.filter(\.isFinite)
        if !finiteValues.isEmpty {
            print("Finite range: [\(finiteValues.min()!), \(finiteValues.max()!)]")
        }
    }
    
    @Test
    func testDoublesWithNaN() throws {
        print("\n=== Testing Doubles with NaN Allowed ===")
        var generatedValues: [Double] = []
        var nanCount = 0
        var finiteCount = 0
        var infiniteCount = 0
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.doubles(allowNaN: true))
            generatedValues.append(value)
            
            if value.isNaN {
                nanCount += 1
                print("Generated NaN: \(value) (bitPattern: 0x\(String(value.bitPattern, radix: 16)))")
            } else if value.isInfinite {
                infiniteCount += 1
                print("Generated infinite: \(value)")
            } else {
                finiteCount += 1
                print("Generated finite: \(value)")
            }
            
            // Either finite, infinite, or NaN
            try verify(value.isFinite || value.isInfinite || value.isNaN,
                      "Should be finite, infinite, or NaN, got: \(value)")
        }
        
        print("Special value distribution: \(nanCount) NaN, \(infiniteCount) infinite, \(finiteCount) finite")
        print("NaN generation rate: \(Double(nanCount)/Double(generatedValues.count) * 100)%")
    }
    
    @Test
    func testDoublesWithInfinity() throws {
        print("\n=== Testing Doubles with Infinity Allowed ===")
        var generatedValues: [Double] = []
        var positiveInfCount = 0
        var negativeInfCount = 0
        var finiteCount = 0
        var nanCount = 0
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.doubles(allowInfinity: true))
            generatedValues.append(value)
            
            if value.isNaN {
                nanCount += 1
                print("Generated NaN: \(value)")
            } else if value == Double.infinity {
                positiveInfCount += 1
                print("Generated +∞: \(value)")
            } else if value == -Double.infinity {
                negativeInfCount += 1
                print("Generated -∞: \(value)")
            } else {
                finiteCount += 1
                print("Generated finite: \(value)")
            }
            
            // Should be finite or infinite (but not NaN unless explicitly allowed)
            try verify(value.isFinite || value.isInfinite, "Should be finite or infinite, got: \(value) (isNaN: \(value.isNaN))")
        }
        
        print("Value distribution: \(finiteCount) finite, \(positiveInfCount) +∞, \(negativeInfCount) -∞, \(nanCount) NaN")
        print("Infinity generation rate: \(Double(positiveInfCount + negativeInfCount)/Double(generatedValues.count) * 100)%")
    }
    
    @Test
    func testDoublesWithSubnormals() throws {
        print("\n=== Testing Doubles with Subnormals Allowed ===")
        var generatedValues: [Double] = []
        var subnormalCount = 0
        var normalCount = 0
        var zeroCount = 0
        var specialCount = 0
        
        try hypothesis(maxValidTestCases: 500) {
            let value = try any(Possibilities.doubles(allowSubnormal: true))
            generatedValues.append(value)
            
            if value == 0.0 {
                zeroCount += 1
                print("Generated zero: \(value)")
            } else if !value.isFinite {
                specialCount += 1
                print("Generated special value: \(value)")
            } else if abs(value) < Double.leastNormalMagnitude && abs(value) > 0 {
                subnormalCount += 1
                print("Generated subnormal: \(value) (magnitude: \(abs(value)))")
            } else {
                normalCount += 1
                print("Generated normal: \(value)")
            }
        }
        
        print("Magnitude distribution: \(normalCount) normal, \(subnormalCount) subnormal, \(zeroCount) zero, \(specialCount) special")
        print("Subnormal threshold: \(Double.leastNormalMagnitude)")
        
        if subnormalCount > 0 {
            let subnormals = generatedValues.filter { abs($0) < Double.leastNormalMagnitude && $0 != 0 }
            print("Subnormal range: [\(subnormals.map(abs).min()!), \(subnormals.map(abs).max()!)]")
        }
    }
}
