import Foundation

private struct Demangler {
    private let demangle: StdlibDemangler
    
    init?() {
        guard let demangler = Self.swift_demangler else {
            return nil
        }
        self.demangle = demangler
    }
    
    func demangle(_ mangled: String) -> String? {
        guard let cString = demangle(mangled, mangled.utf8.count, nil, nil, 0) else {
            return nil
        }
        defer { cString.deallocate() }
        return String(cString: cString)
    }
    
    typealias StdlibDemangler = @convention(c) (
        _ name: UnsafePointer<UInt8>?,
        _ length: Int,
        _ buffer: UnsafeMutablePointer<UInt8>?,
        _ size: UnsafeMutablePointer<Int>?,
        _ flags: UInt32
    ) -> UnsafeMutablePointer<Int8>?
    
    private static var swift_demangler: StdlibDemangler? {
        let stdlib = dlopen(nil, RTLD_NOW)
        guard let symbol = dlsym(stdlib, "swift_demangle") else {
            return nil
        }
        return unsafeBitCast(symbol, to: StdlibDemangler.self)
    }
}

struct Backtrace: Hashable, CustomDebugStringConvertible {
    let stack: [Frame]
    let mangledStack: [String]
    
    init() {
        mangledStack = Thread.callStackSymbols
            .dropLast()
        stack = mangledStack
            .map(Frame.init)
    }
    
    var debugDescription: String {
        """
        === RAW CALL STACK ===
        \(mangledStack.enumerated().map { "\($0): \($1)" }.joined(separator: "\n"))
        === DEMANGLED CALL STACK ===
        \(stack.enumerated().map { "\($0): \($1)" }.joined(separator: "\n"))
        """
    }
}

extension Backtrace {
    struct Frame: Hashable, CustomStringConvertible {
        let index: String
        let symbol: String?
        let offset: String?
        
        var demangledSymbol: String? {
            Demangler().zip(with: symbol).flatMap { demangler, symbol in
                demangler.demangle(symbol)
            }
        }

        init(_ raw: String) {
            let parts = raw.split(separator: /\s/)
            index = String(parts[0])
            symbol = parts.count >= 4 ? String(parts[3]) : nil
            offset = if parts.count >= 6, String(parts[4]).hasPrefix("+") {
                String(parts[4]) + " " + String(parts[5])
            } else {
                nil
            }
        }
        
        var description: String {
            [
                demangledSymbol ?? symbol,
                offset
            ]
                .compactMap(\.self)
                .joined(separator: " ")
        }
    }
}
