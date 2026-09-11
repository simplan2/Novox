using System.Diagnostics;

namespace ElBruno.QwenTTS.Pipeline;

internal sealed class TtsConcurrencyGate : IDisposable
{
    private readonly SemaphoreSlim _semaphore;

    public TtsConcurrencyGate(int maxConcurrency)
    {
        if (maxConcurrency < 1)
            throw new ArgumentOutOfRangeException(nameof(maxConcurrency), maxConcurrency, "Max concurrency must be at least 1.");

        MaxConcurrency = maxConcurrency;
        _semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
    }

    public int MaxConcurrency { get; }

    public async ValueTask<TtsConcurrencyLease> EnterAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        await _semaphore.WaitAsync(cancellationToken);
        stopwatch.Stop();

        return new TtsConcurrencyLease(_semaphore, stopwatch.Elapsed);
    }

    public void Dispose() => _semaphore.Dispose();
}

internal readonly struct TtsConcurrencyLease : IDisposable
{
    private readonly SemaphoreSlim? _semaphore;

    internal TtsConcurrencyLease(SemaphoreSlim semaphore, TimeSpan queueLatency)
    {
        _semaphore = semaphore;
        QueueLatency = queueLatency;
    }

    public TimeSpan QueueLatency { get; }

    public void Dispose() => _semaphore?.Release();
}
