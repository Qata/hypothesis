public
enum Possibilities {
}

public
extension Possibilities {
    static func always<Value>(
        _ value: Value
    ) -> some Possible<Value> {
        AnyPossible { _ in
            value
        }
    }
    
    static func bools() -> some Possible<Bool> {
        AnyPossible {
            try BoundedIntegersPossible(maxValue: 1)
                .provide(from: $0)
            == 1
        }
    }
    
    static func optional<Value>(
        of possible: any Possible<Value>
    ) -> some Possible<Value?> {
        from([
            possible.map(Optional.some),
            always(nil)
        ])
    }
    
    static func integers(min: Int32? = nil, max: Int32? = nil) -> some Possible<Int64> {
        AnyPossible { source in
            switch (min, max) {
            case (nil, nil):
                return try IntegersPossible().provide(from: source)
            case let (nil, max?):
                let base = try IntegersPossible().provide(from: source)
                return numericCast(max) - abs(base)
            case let (min?, nil):
                let base = try IntegersPossible().provide(from: source)
                return numericCast(min) + abs(base)
            case (let min?, let max?):
                guard min <= max else {
                    throw HypothesisError.usageError("min must be <= max")
                }
                let range = Int64(max) - Int64(min)
                let bounded = try BoundedIntegersPossible(maxValue: numericCast(range))
                let offset = try bounded.provide(from: source)
                return numericCast(min) + Int64(offset)
            }
        }
    }
    
    static func codepoints(
        min: UInt32 = 1,
        max: UInt32 = 1_114_111
    ) -> some Possible<UInt32> {
        AnyPossible { source in
            numericCast(
                try BoundedIntegersPossible(
                    range: numericCast(min)...numericCast(max)
                ).provide(
                    from: source
                )
            )
        }
    }
    
    static func strings(
        codepoints codepointsPossible: (any Possible<UInt32>)? = nil,
        minSize: UInt64 = 0,
        maxSize: UInt64 = 10
    ) -> some Possible<String> {
        let codepoints = codepointsPossible ?? codepoints()
        return AnyPossible { source in
            try String(
                String.UnicodeScalarView(
                    arrays(
                        of: AnyPossible(codepoints),
                        minSize: minSize,
                        maxSize: maxSize
                    ).provide(
                        from: source
                    ).compactMap(
                        Unicode.Scalar.init
                    )
                )
            )
        }
    }
    
    static func tuples<each Element: Possible>(
        _ elements: repeat each Element
    ) -> some Possible<(repeat (each Element).Value)> {
        AnyPossible { source in
            (repeat try (each elements).provide(from: source))
        }
    }
    
    static func arrays<Element: Possible>(
        of element: Element,
        minSize: UInt64 = 0,
        maxSize: UInt64 = 10
    ) -> some Possible<[Element.Value]> {
        AnyPossible { source in
            let repeatValues = try CoreRepeatValues(
                minCount: minSize,
                maxCount: maxSize,
                expectedCount: Double(minSize + maxSize) * 0.5
            )
            
            var result: [Element.Value] = []
            
            while try repeatValues.shouldContinue(with: source) {
                let value = try element.provide(from: source)
                result.append(value)
            }
            
            return result
        }
    }
    
    static func from<Value>(
        _ components: [any Possible<Value>]
    ) -> some Possible<Value> {
        AnyPossible { source in
            guard !components.isEmpty else {
                throw HypothesisError.usageError(
                    "Cannot choose from an empty array"
                )
            }
            
            let index = try BoundedIntegersPossible(
                maxValue: numericCast(
                    components.count - 1
                )
            ).provide(from: source)
            
            return try components[numericCast(index)]
                .provide(from: source)
        }
    }
    
    static func from<Value>(
        _ components: [Value]
    ) -> some Possible<Value> {
        AnyPossible { source in
            guard !components.isEmpty else {
                throw HypothesisError.usageError(
                    "Cannot choose from an empty array"
                )
            }
            
            let index = try BoundedIntegersPossible(
                maxValue: numericCast(
                    components.count - 1
                )
            ).provide(from: source)
            
            return components[numericCast(index)]
        }
    }
}

public
extension Possibilities {
    /// A Possible Dictionary with fixed keys and different possible values per key
    static func dictionaries<Key: Hashable, Value>(
        ofShape shape: [Key: any Possible<Value>]
    ) -> some Possible<[Key: Value]> {
        AnyPossible { source in
            var result: [Key: Value] = [:]
            
            for (key, valuePossible) in shape {
                result[key] = try valuePossible.provide(from: source)
            }
            
            return result
        }
    }
    
    static func dictionaries<Key: Hashable, Value>(
        keys: any Possible<Key>,
        values: any Possible<Value>,
        minSize: UInt64 = 0,
        maxSize: UInt64 = 10
    ) -> some Possible<[Key: Value]> {
        AnyPossible { source in
            var result: [Key: Value] = [:]
            
            let repeatValues = try CoreRepeatValues(
                minCount: minSize,
                maxCount: maxSize,
                expectedCount: Double(minSize + maxSize) * 0.5
            )
            
            while try repeatValues.shouldContinue(with: source) {
                let key = try keys.provide(from: source)
                
                if result[key] != nil {
                    try repeatValues.reject()
                } else {
                    let value = try values.provide(from: source)
                    result[key] = value
                }
            }
            
            return result
        }
    }
}
