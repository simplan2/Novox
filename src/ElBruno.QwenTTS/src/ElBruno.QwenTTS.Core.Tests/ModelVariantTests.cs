using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests for QwenModelVariant enum and QwenModelVariantConfig.
/// Validates that each model variant maps to the correct dimensions,
/// repo IDs, and directory layout — without requiring ONNX model files.
/// </summary>
public class ModelVariantTests
{
    // ── Hidden Size ─────────────────────────────────────────────────

    [Fact]
    public void GetHiddenSize_06B_Returns1024()
    {
        Assert.Equal(1024, QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen06B));
    }

    [Fact]
    public void GetHiddenSize_17B_Returns2048()
    {
        Assert.Equal(2048, QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B));
    }

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, 1024)]
    [InlineData(QwenModelVariant.Qwen17B, 2048)]
    public void GetHiddenSize_AllVariants_ReturnExpected(QwenModelVariant variant, int expected)
    {
        Assert.Equal(expected, QwenModelVariantConfig.GetHiddenSize(variant));
    }

    // ── Intermediate Size ───────────────────────────────────────────

    [Fact]
    public void GetIntermediateSize_06B_Returns3072()
    {
        Assert.Equal(3072, QwenModelVariantConfig.GetIntermediateSize(QwenModelVariant.Qwen06B));
    }

    [Fact]
    public void GetIntermediateSize_17B_Returns6144()
    {
        Assert.Equal(6144, QwenModelVariantConfig.GetIntermediateSize(QwenModelVariant.Qwen17B));
    }

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, 3072)]
    [InlineData(QwenModelVariant.Qwen17B, 6144)]
    public void GetIntermediateSize_AllVariants_ReturnExpected(QwenModelVariant variant, int expected)
    {
        Assert.Equal(expected, QwenModelVariantConfig.GetIntermediateSize(variant));
    }

    // ── Intermediate is 3× Hidden ───────────────────────────────────

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B)]
    [InlineData(QwenModelVariant.Qwen17B)]
    public void IntermediateSize_IsThreeTimesHiddenSize(QwenModelVariant variant)
    {
        var hidden = QwenModelVariantConfig.GetHiddenSize(variant);
        var intermediate = QwenModelVariantConfig.GetIntermediateSize(variant);
        Assert.Equal(hidden * 3, intermediate);
    }

    // ── HuggingFace Repo IDs ────────────────────────────────────────

    [Fact]
    public void GetRepoId_06B_MatchesDefaultRepoId()
    {
        Assert.Equal(ModelDownloader.DefaultRepoId,
            QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen06B));
    }

    [Fact]
    public void GetRepoId_17B_ContainsCorrectModelName()
    {
        var repoId = QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen17B);
        Assert.Contains("1.7B", repoId);
        Assert.StartsWith("elbruno/", repoId);
        Assert.Contains("ONNX", repoId);
    }

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, "0.6B")]
    [InlineData(QwenModelVariant.Qwen17B, "1.7B")]
    public void GetRepoId_ContainsVariantSizeName(QwenModelVariant variant, string sizeName)
    {
        var repoId = QwenModelVariantConfig.GetRepoId(variant);
        Assert.Contains(sizeName, repoId);
    }

    [Fact]
    public void GetRepoId_VariantsHaveDifferentRepoIds()
    {
        var repo06B = QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen06B);
        var repo17B = QwenModelVariantConfig.GetRepoId(QwenModelVariant.Qwen17B);
        Assert.NotEqual(repo06B, repo17B);
    }

    // ── Model Subdirectories ────────────────────────────────────────

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, "0.6B")]
    [InlineData(QwenModelVariant.Qwen17B, "1.7B")]
    public void GetModelSubDir_ReturnsExpected(QwenModelVariant variant, string expected)
    {
        Assert.Equal(expected, QwenModelVariantConfig.GetModelSubDir(variant));
    }

    [Fact]
    public void GetDefaultModelDir_VariantsHaveDifferentPaths()
    {
        var dir06B = QwenModelVariantConfig.GetDefaultModelDir(QwenModelVariant.Qwen06B);
        var dir17B = QwenModelVariantConfig.GetDefaultModelDir(QwenModelVariant.Qwen17B);
        Assert.NotEqual(dir06B, dir17B);
    }

    [Fact]
    public void GetDefaultModelDir_BothUnderSharedRoot()
    {
        var root = ModelDownloader.DefaultModelDir;
        var dir06B = QwenModelVariantConfig.GetDefaultModelDir(QwenModelVariant.Qwen06B);
        var dir17B = QwenModelVariantConfig.GetDefaultModelDir(QwenModelVariant.Qwen17B);
        // 0.6B uses the legacy root path for backward compat
        Assert.Equal(root, dir06B);
        Assert.StartsWith(root, dir17B);
    }

    [Fact]
    public void GetDefaultModelDir_ContainsSubDir()
    {
        var dir = QwenModelVariantConfig.GetDefaultModelDir(QwenModelVariant.Qwen17B);
        Assert.EndsWith("1.7B", dir);
    }

    // ── Default Variant ─────────────────────────────────────────────

    [Fact]
    public void DefaultVariant_Is06B()
    {
        Assert.Equal(QwenModelVariant.Qwen06B, QwenModelVariantConfig.Default);
    }

    [Fact]
    public void DefaultVariant_HiddenSize_Is1024()
    {
        Assert.Equal(1024, QwenModelVariantConfig.GetHiddenSize(QwenModelVariantConfig.Default));
    }

    [Fact]
    public void DefaultVariant_RepoId_MatchesDownloaderDefault()
    {
        Assert.Equal(ModelDownloader.DefaultRepoId,
            QwenModelVariantConfig.GetRepoId(QwenModelVariantConfig.Default));
    }

    // ── Invalid Variant Handling ────────────────────────────────────

    [Fact]
    public void GetHiddenSize_InvalidVariant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => QwenModelVariantConfig.GetHiddenSize((QwenModelVariant)99));
    }

    [Fact]
    public void GetIntermediateSize_InvalidVariant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => QwenModelVariantConfig.GetIntermediateSize((QwenModelVariant)99));
    }

    [Fact]
    public void GetRepoId_InvalidVariant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => QwenModelVariantConfig.GetRepoId((QwenModelVariant)99));
    }

    [Fact]
    public void GetModelSubDir_InvalidVariant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => QwenModelVariantConfig.GetModelSubDir((QwenModelVariant)99));
    }

    // ── GetAllVariants ──────────────────────────────────────────────

    [Fact]
    public void GetAllVariants_ContainsBothVariants()
    {
        var all = QwenModelVariantConfig.GetAllVariants();
        Assert.Contains(QwenModelVariant.Qwen06B, all);
        Assert.Contains(QwenModelVariant.Qwen17B, all);
        Assert.Equal(2, all.Length);
    }

    [Fact]
    public void GetAllVariants_AllHaveValidConfig()
    {
        foreach (var variant in QwenModelVariantConfig.GetAllVariants())
        {
            Assert.True(QwenModelVariantConfig.GetHiddenSize(variant) > 0);
            Assert.True(QwenModelVariantConfig.GetIntermediateSize(variant) > 0);
            Assert.False(string.IsNullOrEmpty(QwenModelVariantConfig.GetRepoId(variant)));
            Assert.False(string.IsNullOrEmpty(QwenModelVariantConfig.GetModelSubDir(variant)));
        }
    }

    // ── Enum Values ─────────────────────────────────────────────────

    [Fact]
    public void QwenModelVariant_06B_IsZero()
    {
        // Ensures 0.6B is the default when default(QwenModelVariant) is used
        Assert.Equal(0, (int)QwenModelVariant.Qwen06B);
    }

    [Fact]
    public void QwenModelVariant_DefaultEnum_Is06B()
    {
        // C# default for an enum is 0 — confirming this maps to 0.6B
        Assert.Equal(QwenModelVariant.Qwen06B, default(QwenModelVariant));
    }
}
