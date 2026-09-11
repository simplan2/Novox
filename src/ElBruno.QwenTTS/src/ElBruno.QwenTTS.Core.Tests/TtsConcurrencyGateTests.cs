using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

public class TtsConcurrencyGateTests
{
    [Fact]
    public void Constructor_WithInvalidConcurrency_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TtsConcurrencyGate(0));
    }

    [Fact]
    public async Task EnterAsync_CancelledWhileWaiting_Throws()
    {
        using var gate = new TtsConcurrencyGate(1);
        var firstLease = await gate.EnterAsync();

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            _ = await gate.EnterAsync(cts.Token);
        });

        firstLease.Dispose();
    }

    [Fact]
    public async Task EnterAsync_ReportsQueueLatency()
    {
        using var gate = new TtsConcurrencyGate(1);
        var firstLease = await gate.EnterAsync();

        var releaseTask = Task.Run(async () =>
        {
            await Task.Delay(75);
            firstLease.Dispose();
        });

        var secondLease = await gate.EnterAsync();
        try
        {
            Assert.True(secondLease.QueueLatency >= TimeSpan.FromMilliseconds(50),
                $"Expected queue latency >= 50 ms, got {secondLease.QueueLatency.TotalMilliseconds:F1} ms.");
        }
        finally
        {
            secondLease.Dispose();
            await releaseTask;
        }
    }
}
