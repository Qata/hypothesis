import Testing
import Foundation
@testable import Hypothesis

@Suite
struct DistributionTests {
    
    // MARK: - Custom Distribution Tests
    
    /// Custom distribution for testing protocol conformance
    struct Dice: ConjectureDistribution {
        let sides: UInt64
        
        func provide(from source: CoreDataSource) throws -> UInt64 {
            let bounded = try CoreBoundedIntegers(maxValue: sides - 1)
            return try bounded.provide(from: source) + 1
        }
    }
    
    @Test
    func testCustomDistribution() throws {
        let engine = try CoreEngine(
            name: "custom_distribution",
            seed: 42,
            maxExamples: 100
        )
        
        let d6 = Dice(sides: 6)
        let d20 = Dice(sides: 20)
        
        var d6Results: [UInt64] = []
        var d20Results: [UInt64] = []
        
        while let source = try engine.newSource() {
            do {
                let roll6 = try source.draw(d6)
                let roll20 = try source.draw(d20)
                
                d6Results.append(roll6)
                d20Results.append(roll20)
                
                #expect(1...6 ~= roll6)
                #expect(1...20 ~= roll20)
                
                try engine.finish(source, .valid)
            } catch CoreError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
        
        // Check distribution properties
        let unique6 = Set(d6Results).count
        let unique20 = Set(d20Results).count
        
        #expect(unique6 >= 4, "Should see most dice faces with 100 rolls")
        #expect(unique20 >= 10, "Should see many d20 faces with 100 rolls")
    }
    
    // MARK: - Composite Distribution Tests
    
    struct Point: Equatable {
        let x: Int64
        let y: Int64
    }
    
    struct Points: ConjectureDistribution {
        let xRange: ClosedRange<Int64>
        let yRange: ClosedRange<Int64>
        
        func provide(from source: CoreDataSource) throws -> Point {
            // Note: This is a workaround since we don't have signed bounded integers
            let xBase = try CoreBoundedIntegers(
                maxValue: UInt64(xRange.upperBound - xRange.lowerBound)
            ).provide(from: source)
            let yBase = try CoreBoundedIntegers(
                maxValue: UInt64(yRange.upperBound - yRange.lowerBound)
            ).provide(from: source)
            
            return Point(
                x: xRange.lowerBound + Int64(xBase),
                y: yRange.lowerBound + Int64(yBase)
            )
        }
    }
    
    @Test
    func testCompositeDistribution() throws {
        let engine = try CoreEngine(
            name: "composite_distribution",
            seed: 12345,
            maxExamples: 50
        )
        
        let points = Points(xRange: -10...10, yRange: -10...10)
        
        var quadrants = [Int: Int]()
        
        while let source = try engine.newSource() {
            do {
                let point = try source.draw(points)
                
                #expect(points.xRange.contains(point.x))
                #expect(points.yRange.contains(point.y))
                
                // Track which quadrant
                let quadrant: Int
                if point.x >= 0 && point.y >= 0 {
                    quadrant = 1
                } else if point.x < 0 && point.y >= 0 {
                    quadrant = 2
                } else if point.x < 0 && point.y < 0 {
                    quadrant = 3
                } else {
                    quadrant = 4
                }
                
                quadrants[quadrant, default: 0] += 1
                
                try engine.finish(source, .valid)
            } catch CoreError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
        
        // Should have points in multiple quadrants
        #expect(quadrants.count >= 3, "Points should be distributed across quadrants")
    }
    
    // MARK: - Generic Distribution Function Tests
    
    func collectValues<D: ConjectureDistribution>(
        _ distribution: D,
        count: Int,
        seed: UInt64 = 99999
    ) throws -> [D.Value] {
        let engine = try CoreEngine(
            name: "collect_values",
            seed: seed,
            maxExamples: UInt64(count)
        )
        
        var values: [D.Value] = []
        
        while let source = try engine.newSource() {
            do {
                try values.append(source.draw(distribution))
                try engine.finish(source, .valid)
            } catch CoreError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
        
        return values
    }
    
    @Test
    func testGenericCollectionFunction() throws {
        // Test with integers
        let integers = try collectValues(
            CoreIntegers.unbounded(),
            count: 10
        )
        #expect(integers.count == 10)
        
        // Test with bounded integers
        let bounded = try collectValues(
            CoreBoundedIntegers(maxValue: 5),
            count: 20
        )
        #expect(bounded.count == 20)
        #expect(bounded.allSatisfy { $0 <= 5 })
        
        // Test with custom distribution
        let dice = try collectValues(Dice(sides: 6), count: 30)
        #expect(dice.count == 30)
        #expect(dice.allSatisfy { $0 >= 1 && $0 <= 6 })
    }
    
    // MARK: - Distribution Transformation Tests
    
    struct TransformedDistribution<Base: ConjectureDistribution, Output>: ConjectureDistribution {
        let base: Base
        let transform: (Base.Value) throws -> Output
        
        func provide(from source: CoreDataSource) throws -> Output {
            let baseValue = try base.provide(from: source)
            return try transform(baseValue)
        }
    }
    
    @Test
    func testDistributionTransformation() throws {
        // Create a distribution of even numbers
        let evens = TransformedDistribution(
            base: try CoreBoundedIntegers(maxValue: 50),
            transform: { $0 * 2 }
        )

        let values = try collectValues(evens, count: 20)

        #expect(values.allSatisfy { $0 % 2 == 0 }, "All values should be even")
        #expect(values.allSatisfy { $0 <= 100 }, "All values should be <= 100")

        // Create a distribution of strings from integers
        let strings = TransformedDistribution(
            base: try CoreBoundedIntegers(maxValue: 99),
            transform: {
                String($0, radix: 10, uppercase: true)
            }
        )

        #expect(
            try collectValues(strings, count: 10)
                .allSatisfy { Int($0) != nil }
        )
    }
    
    // MARK: - Array Building Tests
    
    struct Arrays<Element: ConjectureDistribution>: ConjectureDistribution {
        typealias Value = [Element.Value]
        
        let element: Element
        let sizeRange: ClosedRange<UInt64>
        
        func provide(from source: CoreDataSource) throws -> [Element.Value] {
            let repeat_ = try CoreRepeatValues.range(sizeRange)
            var result: [Element.Value] = []
            
            while try repeat_.shouldContinue(with: source) {
                result.append(try source.draw(element))
            }
            
            return result
        }
    }
    
    @Test
    func testArrayDistribution() throws {
        let arrays = Arrays(
            element: try CoreBoundedIntegers(maxValue: 10),
            sizeRange: 0...5
        )
        
        let values = try collectValues(arrays, count: 50)
        
        // Check size constraints
        #expect(values.allSatisfy { $0.count >= 0 && $0.count <= 5 })
        
        // Check element constraints
        #expect(values.allSatisfy { array in
            array.allSatisfy { $0 <= 10 }
        })
        
        // Check we get various sizes
        let sizes = Set(values.map { $0.count })
        #expect(sizes.count >= 3, "Should generate arrays of various sizes")
    }
    
    // MARK: - Edge Case Distribution Tests
    
    @Test
    func testSingleValueDistribution() throws {
        // Distribution that always returns the same value
        struct Constant: ConjectureDistribution {
            let value: Int
            
            func provide(from source: CoreDataSource) throws -> Int {
                // Still consume some randomness to advance the source
                _ = try source.bits(8)
                return value
            }
        }
        
        let constant = Constant(value: 42)
        let values = try collectValues(constant, count: 10)
        
        #expect(values.allSatisfy { $0 == 42 })
    }
    
    // MARK: - Performance Tests
    
    @Test
    func testLargeArrayGeneration() throws {
        let engine = try CoreEngine(
            name: "large_array_test",
            seed: 54321,
            maxExamples: 10
        )
        
        while let source = try engine.newSource() {
            do {
                // Generate a potentially large array
                let repeat_ = try CoreRepeatValues.range(0...1000)
                let bounded = try CoreBoundedIntegers(maxValue: 255)
                
                var array: [UInt64] = []
                while try repeat_.shouldContinue(with: source) {
                    array.append(try source.draw(bounded))
                }
                
                #expect(array.count <= 1000)
                
                try engine.finish(source, .valid)
            } catch CoreError.dataOverflow {
                try engine.finish(source, .overflow)
            }
        }
    }
}
