// MARK: - Zipping

// TODO: - This code is gross and hacky but at least it works with Swift 5.9's handling of variadic generics.
public
func zipOptional<each Value>(
    _ value: repeat (each Value)?
) -> (repeat each Value)? {
    try? zipImplementation(repeat each value) as (repeat each Value)
}

/// The underlying implementation of `zipOptional`.
private
func zipImplementation<each Value>(
    _ value: repeat (each Value)?
) throws -> (repeat each Value) {
    repeat try throwIfNil(each value)
    return (repeat (each value)!)
}

private
struct NilError: Error {}

private
func throwIfNil<Value>(_ value: Value?) throws {
    if value == nil { throw NilError() }
}

public
extension Optional {
    func zip<each Value>(
        with optionals: repeat Optional<each Value>
    ) -> Optional<(Wrapped, repeat each Value)> {
        zipOptional(self, repeat each optionals)
    }
}

// MARK: - Filtering

public
extension Optional {
    func filter(_ shouldKeep: (Wrapped) -> Bool) -> Self {
        switch self {
        case let .some(wrapped) where shouldKeep(wrapped):
            wrapped
        case .some, .none:
            nil
        }
    }

    func filter<T>(_ keyPath: KeyPath<Wrapped, T>, _ shouldKeep: (T) -> Bool) -> Self {
        filter { shouldKeep($0[keyPath: keyPath]) }
    }
}

// MARK: - Unwrapping

public
extension Optional {
    func unwrap(throwing error: Error) throws -> Wrapped {
        switch self {
        case let .some(wrapped):
            wrapped
        case .none:
            throw error
        }
    }
}
