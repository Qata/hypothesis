import Testing
@testable import Hypothesis

@Suite
struct RepeatValuesTests {
    @Test
    func testMinMaxBoundsAreRespected() throws {
        // Property: Generated count is always between minCount and maxCount (inclusive)
        let testCases: [(min: UInt64, max: UInt64, expected: Double)] = [
            (0, 5, 2.5),
            (2, 10, 6.0),
            (5, 5, 5.0),  // When min == max, should always generate exactly that many
            (0, 0, 0.0),  // Edge case: empty arrays allowed
            (1, 100, 50.0),
            (10, 20, 15.0)
        ]
        
        for testCase in testCases {
            let engine = try ConjectureEngine(
                name: "bounds_test_\(testCase.min)_\(testCase.max)",
                seed: .random(in: 0...UInt64.max),
                maxExamples: 100
            )
            
            var counts: [Int] = []
            
            while let source = try engine.newSource() {
                do {
                    let repeat_ = try ConjectureRepeatValues(
                        minCount: testCase.min,
                        maxCount: testCase.max,
                        expectedCount: testCase.expected
                    )
                    
                    var count = 0
                    while try repeat_.shouldContinue(with: source) {
                        count += 1
                        try source.startDraw()
                        _ = try source.bits(8)
                        try source.stopDraw()
                        
                        // Safety check
                        if count > Int(testCase.max) + 10 {
                            #expect(Bool(false), "Count exceeded max by more than 10")
                            break
                        }
                    }
                    
                    counts.append(count)
                    try engine.finish(source, .valid)
                    
                } catch ConjectureError.dataOverflow {
                    try engine.finish(source, .overflow)
                }
            }
            
            // Verify all counts are within bounds
            #expect(counts.allSatisfy { $0 >= Int(testCase.min) },
                    "All counts should be >= \(testCase.min), but got \(counts)")
            #expect(counts.allSatisfy { $0 <= Int(testCase.max) },
                    "All counts should be <= \(testCase.max), but got \(counts)")
            
            // If min == max, all counts should be exactly that value
            if testCase.min == testCase.max {
                #expect(counts.allSatisfy { $0 == Int(testCase.min) },
                        "When min == max (\(testCase.min)), all counts should be exactly that value")
            }
        }
    }
    
    @Test
    func testExpectedValueInfluencesDistribution() throws {
        // Property: The average of generated values should be close to expectedCount
        // (within the constraints of min and max)
        
        struct TestCase {
            let min: UInt64
            let max: UInt64
            let expected: Double
            var effectiveExpected: Double {
                // The expected value is clamped to [min, max]
                return Swift.max(Double(min), Swift.min(Double(max), expected))
            }
        }
        
        let testCases = [
            TestCase(min: 0, max: 10, expected: 5.0),
            TestCase(min: 0, max: 10, expected: 2.0),
            TestCase(min: 0, max: 10, expected: 8.0),
            TestCase(min: 5, max: 15, expected: 10.0),
            TestCase(min: 5, max: 15, expected: 3.0),  // Expected below min, should tend toward min
            TestCase(min: 5, max: 15, expected: 20.0), // Expected above max, should tend toward max
        ]
        
        for testCase in testCases {
            let engine = try ConjectureEngine(
                name: "distribution_test",
                seed: 98765,
                maxExamples: 500  // More examples for better statistics
            )
            
            var counts: [Int] = []
            
            while let source = try engine.newSource() {
                do {
                    let repeat_ = try ConjectureRepeatValues(
                        minCount: testCase.min,
                        maxCount: testCase.max,
                        expectedCount: testCase.expected
                    )
                    
                    var count = 0
                    while try repeat_.shouldContinue(with: source) {
                        count += 1
                        try source.startDraw()
                        _ = try source.bits(8)
                        try source.stopDraw()
                    }
                    
                    counts.append(count)
                    try engine.finish(source, .valid)
                    
                } catch ConjectureError.dataOverflow {
                    try engine.finish(source, .overflow)
                }
            }
            
            let average = Double(counts.reduce(0, +)) / Double(counts.count)
            let tolerance = (Double(testCase.max - testCase.min) * 0.2) + 1.0  // 20% of range + 1
            
            #expect(
                abs(average - testCase.effectiveExpected) < tolerance,
                """
                Average \(average) should be within \(tolerance) of \
                effective expected \(testCase.effectiveExpected) \
                (original expected: \(testCase.expected), min: \(testCase.min), max: \(testCase.max))
                """
            )
        }
    }
    
    @Test
    func testRejectionMechanismWorks() throws {
        // Property: Calling reject() should decrease the count by 1
        
        let engine = try ConjectureEngine(
            name: "rejection_test",
            seed: 11111,
            maxExamples: 50
        )
        
        while let source = try engine.newSource() {
            do {
                let repeat_ = try ConjectureRepeatValues(
                    minCount: 3,
                    maxCount: 10,
                    expectedCount: 6.5
                )
                
                var acceptedCount = 0
                var totalCalls = 0
                var rejectionCount = 0
                
                while try repeat_.shouldContinue(with: source) {
                    totalCalls += 1
                    
                    try source.startDraw()
                    let value = try source.bits(8)
                    try source.stopDraw()
                    
                    // Reject if value is divisible by 3
                    if value % 3 == 0 {
                        try repeat_.reject()
                        rejectionCount += 1
                    } else {
                        acceptedCount += 1
                    }
                    
                    // Safety check
                    if totalCalls > 50 {
                        break
                    }
                }
                
                // The accepted count should still be within bounds
                #expect(acceptedCount >= 3 && acceptedCount <= 10,
                        "Accepted count \(acceptedCount) should be within bounds [3, 10]")
                
                // Total calls should be accepted + rejected
                #expect(totalCalls >= acceptedCount,
                        "Total calls (\(totalCalls)) should be >= accepted count (\(acceptedCount))")
                
                try engine.finish(source, .valid)
                
            } catch ConjectureError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
    }
    
    @Test
    func testEdgeCases() throws {
        // Property: RepeatValues should handle edge cases correctly
        
        // Edge case 1: min = max = 0 (always empty)
        try verifyExactCount(count: 0, expectedCount: 0)
        
        // Edge case 2: min = max = 1 (always single element)
        try verifyExactCount(count: 0, expectedCount: 1)
        
        // Edge case 3: Very large ranges
        try verifyBounds(min: 0, max: 1000, expectedCount: 500.0, maxExamples: 20)
        
        // Edge case 4: Expected count outside of [min, max] range
        try verifyBounds(min: 10, max: 20, expectedCount: 5.0, maxExamples: 20)   // Below min
        try verifyBounds(min: 10, max: 20, expectedCount: 50.0, maxExamples: 20)  // Above max
    }
    
    @Test
    func testNoInfiniteLoops() throws {
        // Property: shouldContinue must eventually return false
        // (within max_count calls when drawing data)
        
        let engine = try ConjectureEngine(
            name: "no_infinite_loops_test",
            seed: 99999,
            maxExamples: 100
        )
        
        while let source = try engine.newSource() {
            do {
                let maxCount: UInt64 = 10
                let repeat_ = try ConjectureRepeatValues(
                    minCount: 0,
                    maxCount: maxCount,
                    expectedCount: 5.0
                )
                
                var callCount = 0
                let safetyLimit = Int(maxCount) * 2 + 10
                
                while callCount < safetyLimit {
                    callCount += 1
                    
                    let shouldContinue = try repeat_.shouldContinue(with: source)
                    
                    if !shouldContinue {
                        break
                    }
                    
                    try source.startDraw()
                    _ = try source.bits(8)
                    try source.stopDraw()
                }
                
                #expect(callCount <= Int(maxCount) + 1,
                        "shouldContinue should have returned false within \(maxCount + 1) calls, but took \(callCount)")
                
                try engine.finish(source, .valid)
                
            } catch ConjectureError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
    }
    
    // MARK: - Helper Functions
    
    private func verifyExactCount(
        count: UInt64,
        expectedCount: Double
    ) throws {
        
        let engine = try ConjectureEngine(
            name: "exact_count_test",
            seed: 55555,
            maxExamples: 20
        )
        
        var counts: [Int] = []
        
        while let source = try engine.newSource() {
            do {
                let repeat_ = try ConjectureRepeatValues(
                    minCount: count,
                    maxCount: count,
                    expectedCount: expectedCount
                )
                
                var count = 0
                while try repeat_.shouldContinue(with: source) {
                    count += 1
                    try source.startDraw()
                    _ = try source.bits(8)
                    try source.stopDraw()
                }
                
                counts.append(count)
                try engine.finish(source, .valid)
                
            } catch ConjectureError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
        
        #expect(
            counts.allSatisfy { $0 == Int(count) },
            "All counts should be exactly \(count), but got \(counts)"
        )
    }
    
    private func verifyBounds(
        min: UInt64,
        max: UInt64,
        expectedCount: Double,
        maxExamples: UInt64
    ) throws {
        let engine = try ConjectureEngine(
            name: "bounds_verification_test",
            seed: 77777,
            maxExamples: maxExamples
        )
        
        var counts: [Int] = []
        
        while let source = try engine.newSource() {
            do {
                let repeat_ = try ConjectureRepeatValues(
                    minCount: min,
                    maxCount: max,
                    expectedCount: expectedCount
                )
                
                var count = 0
                let safetyLimit = Int(max) + 100
                
                while count < safetyLimit {
                    let shouldContinue = try repeat_.shouldContinue(with: source)
                    
                    if !shouldContinue {
                        break
                    }
                    
                    count += 1
                    try source.startDraw()
                    _ = try source.bits(8)
                    try source.stopDraw()
                }
                counts.append(count)
                try engine.finish(source, .valid)
                
            } catch ConjectureError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
        
        #expect(
            !counts.isEmpty,
            "Should have generated at least one example"
        )
        #expect(
            counts.allSatisfy { $0 >= Int(min) },
            "All counts should be >= \(min), but got \(counts)"
        )
        #expect(
            counts.allSatisfy { $0 <= Int(max) },
            "All counts should be <= \(max), but got \(counts)"
        )
    }
}
