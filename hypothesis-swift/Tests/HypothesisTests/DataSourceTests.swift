import Testing
@testable import Hypothesis

@Suite
struct DataSourceTests {
    @Test
    func testDataSourceConsumption() throws {
        let engine = try ConjectureEngine(
            name: "consumption_test",
            seed: 12345,
            maxExamples: 1
        )
        
        guard let source = try engine.newSource() else {
            Issue.record("Expected at least one source")
            return
        }
        
        // First consumption should succeed
        let handle = try source.consume()
        #expect(handle != nil)
        
        // Second consumption should fail
        #expect(throws: ConjectureError.dataSourceAlreadyConsumed) {
            _ = try source.consume()
        }
        
        // Operations after consumption should fail
        #expect(throws: ConjectureError.dataSourceAlreadyConsumed) {
            _ = try source.bits(8)
        }
        
        #expect(throws: ConjectureError.dataSourceAlreadyConsumed) {
            try source.startDraw()
        }
    }
}
