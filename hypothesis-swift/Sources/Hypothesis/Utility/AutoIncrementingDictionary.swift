struct AutoIncrementingDictionary<Key: Hashable> {
    private var storage: [Key: UInt64] = [:]
    private var nextValue: UInt64 = 0
    
    func hasKey(_ key: Key) -> Bool {
        storage.keys.contains(key)
    }
    
    func sorted() -> [(key: Key, value: UInt64)] {
        storage.sorted(by: { $0.value < $1.value })
    }
    
    subscript(key: Key) -> UInt64 {
        mutating get {
            if let existing = storage[key] {
                return existing
            } else {
                defer { nextValue += 1 }
                storage[key] = nextValue
                return nextValue
            }
        }
    }
}
