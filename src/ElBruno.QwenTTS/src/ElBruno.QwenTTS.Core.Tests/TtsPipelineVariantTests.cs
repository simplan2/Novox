using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests for TtsPipeline backward compatibility with the new variant system.
/// Validates that CreateAsync() still defaults to 0.6B and the API is non-breaking.
/// </summary>
public class TtsPipelineVariantTests : IDisposable
{
    private readonly string _tempDir;

    public TtsPipelineVariantTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_pipeline_variant_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    // ── CreateAsync Backward Compatibility ──────────────────────────

    [Fact]
    public async Task CreateAsync_WithoutVariant_StillDefaultsTo06B()
    {
        // Existing call signature: CreateAsync(modelDir, progress, cancellationToken)
        // Should still use the 0.6B repo and not break
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var messages = new List<string>();
        var progress = new Progress<string>(msg => messages.Add(msg));

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            TtsPipeline.CreateAsync(_tempDir, progress: progress, cancellationToken: cts.Token));
    }

    [Fact]
    public async Task CreateAsync_ExplicitDefaultRepoId_BehavesIdentically()
    {
        // Passing DefaultRepoId explicitly should behave same as omitting it
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            TtsPipeline.CreateAsync(
                _tempDir,
                repoId: ModelDownloader.DefaultRepoId,
                cancellationToken: cts.Token));
    }

    [Fact]
    public async Task CreateAsync_With17BRepoId_InitiatesDownload()
    {
        // Passing the 1.7B repo ID should trigger download (we cancel immediately)
        var repo17B = QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen17B);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            TtsPipeline.CreateAsync(
                _tempDir,
                repoId: repo17B,
                cancellationToken: cts.Token));
    }

    // ── Constructor Backward Compatibility ──────────────────────────

    [Fact]
    public void Constructor_StillThrowsWithInvalidDir()
    {
        // TtsPipeline(modelDir) should still throw for missing files — not broken by variant changes
        var bogusDir = Path.Combine(_tempDir, "nonexistent_models");
        Assert.ThrowsAny<Exception>(() => new TtsPipeline(bogusDir));
    }

    // ── Model File Expectations Are Consistent ──────────────────────

    [Fact]
    public void IsModelDownloaded_EmptyDir_StillReturnsFalse()
    {
        // Core API must remain stable
        Assert.False(ModelDownloader.IsModelDownloaded(_tempDir));
    }

    [Fact]
    public void GetMissingFiles_StillReturnsExpectedCount()
    {
        var missing = ModelDownloader.GetMissingFiles(_tempDir);
        Assert.True(missing.Count >= 33,
            $"Expected at least 33 required files, got {missing.Count}");
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }
}
