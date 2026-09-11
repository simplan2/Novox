using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests that ModelDownloader correctly handles variant-specific repo IDs and directories.
/// Also validates backward compatibility — existing APIs still work without variant parameter.
/// </summary>
public class ModelVariantDownloaderTests : IDisposable
{
    private readonly string _tempDir;

    public ModelVariantDownloaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_variant_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    // ── Backward Compatibility ──────────────────────────────────────

    [Fact]
    public void DefaultRepoId_StillPointsTo06B()
    {
        // ModelDownloader.DefaultRepoId must remain the 0.6B repo for backward compat
        Assert.Equal(
            ModelDownloader.DefaultRepoId,
            QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen06B));
    }

    [Fact]
    public void IsModelReady_StillWorksWithoutVariant()
    {
        // Existing code calling IsModelReady(dir) should still compile and work
        Assert.False(ModelDownloader.IsModelReady(_tempDir));
    }

    [Fact]
    public void GetMissingFiles_StillWorksWithoutVariant()
    {
        // Existing code calling GetMissingFiles(dir) should still compile and work
        var missing = ModelDownloader.GetMissingFiles(_tempDir);
        Assert.NotEmpty(missing);
    }

    [Fact]
    public async Task DownloadModelAsync_StillAcceptsDefaultRepoId()
    {
        // Existing code passing DefaultRepoId should still work
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            ModelDownloader.DownloadModelAsync(_tempDir, ModelDownloader.DefaultRepoId, cancellationToken: cts.Token));
    }

    // ── Variant-Specific Repo IDs ───────────────────────────────────

    [Fact]
    public async Task DownloadModelAsync_CanAccept17BRepoId()
    {
        // The 1.7B repo ID should be accepted without error (cancel before actual download)
        var repo17B = QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen17B);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            ModelDownloader.DownloadModelAsync(_tempDir, repo17B, cancellationToken: cts.Token));
    }

    // ── Variant-Specific Directories ────────────────────────────────

    [Fact]
    public void VariantDirectories_AreSeparate()
    {
        // Create dirs for each variant and verify they don't overlap
        var dir06B = Path.Combine(_tempDir, QwenModelVariantConfig.GetModelSubDir(QwenModelVariant.Qwen06B));
        var dir17B = Path.Combine(_tempDir, QwenModelVariantConfig.GetModelSubDir(QwenModelVariant.Qwen17B));

        Directory.CreateDirectory(dir06B);
        Directory.CreateDirectory(dir17B);

        Assert.NotEqual(dir06B, dir17B);
        Assert.True(Directory.Exists(dir06B));
        Assert.True(Directory.Exists(dir17B));
    }

    [Fact]
    public void VariantDirectories_DontMixFiles()
    {
        // Write a marker file in 0.6B dir — it should not appear in 1.7B dir
        var dir06B = Path.Combine(_tempDir, QwenModelVariantConfig.GetModelSubDir(QwenModelVariant.Qwen06B));
        var dir17B = Path.Combine(_tempDir, QwenModelVariantConfig.GetModelSubDir(QwenModelVariant.Qwen17B));

        Directory.CreateDirectory(dir06B);
        Directory.CreateDirectory(dir17B);

        File.WriteAllText(Path.Combine(dir06B, "marker.txt"), "0.6B");

        Assert.True(File.Exists(Path.Combine(dir06B, "marker.txt")));
        Assert.False(File.Exists(Path.Combine(dir17B, "marker.txt")));
    }

    [Fact]
    public void IsModelReady_PerVariant_IndependentState()
    {
        var dir06B = Path.Combine(_tempDir, QwenModelVariantConfig.GetModelSubDir(QwenModelVariant.Qwen06B));
        var dir17B = Path.Combine(_tempDir, QwenModelVariantConfig.GetModelSubDir(QwenModelVariant.Qwen17B));
        Directory.CreateDirectory(dir06B);
        Directory.CreateDirectory(dir17B);

        // Populate 0.6B with all stub files
        var missing06B = ModelDownloader.GetMissingFiles(dir06B);
        foreach (var file in missing06B)
        {
            var path = Path.Combine(dir06B, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, "stub");
        }

        // 0.6B should be ready, 1.7B should NOT
        Assert.True(ModelDownloader.IsModelReady(dir06B));
        Assert.False(ModelDownloader.IsModelReady(dir17B));
    }

    // ── QwenTtsOptions Variant Integration ──────────────────────────

    [Fact]
    public void QwenTtsOptions_DefaultHuggingFaceRepo_IsNull()
    {
        var opts = new QwenTtsOptions();
        // HuggingFaceRepo defaults to null — variant determines the repo at resolve time
        Assert.Null(opts.HuggingFaceRepo);
        Assert.Equal(QwenModelVariant.Qwen06B, opts.ModelVariant);
    }

    [Fact]
    public void QwenTtsOptions_CanOverrideRepoFor17B()
    {
        var opts = new QwenTtsOptions
        {
            HuggingFaceRepo = QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen17B)
        };
        Assert.Contains("1.7B", opts.HuggingFaceRepo);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }
}
