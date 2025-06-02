struct BacktracedError: Error {
    let backtrace = Backtrace()
    let underlying: Error
}
