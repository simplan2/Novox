using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Edge case tests for QwenModelVariant and QwenModelVariantConfig.
/// Covers invalid variant handling, GetDefaultModelDir edge cases,
/// ResolveForVariant, and QwenTtsOptions variant integration.
/// </summary>
public class ModelVariantEdgeCaseTests
{
    // ── GetDefaultModelDir Invalid Variant ──────────────────────────

    [Fact]
    public void GetDefaultModelDir_InvalidVariant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => QwenModelVariantConfig.GetDefaultModelDir((QwenModelVariant)99));
    }

    // ── ResolveForVariant ───────────────────────────────────────────

    [Fact]
    public void ResolveForVariant_06B_NullOverrides_UsesDefaults()
    {
        var (dir, repo) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen06B);
        Assert.Equal(ModelDownloader.DefaultModelDir, dir);
        Assert.Equal(ModelDownloader.DefaultRepoId, repo);
    }

    [Fact]
    public void ResolveForVariant_17B_NullOverrides_UsesVariantDefaults()
    {
        var (dir, repo) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen17B);
        Assert.EndsWith("1.7B", dir);
        Assert.Contains("1.7B", repo);
        Assert.NotEqual(ModelDownloader.DefaultModelDir, dir);
        Assert.NotEqual(ModelDownloader.DefaultRepoId, repo);
    }

    [Fact]
    public void ResolveForVariant_CustomModelDir_OverridesDefault()
    {
        var customDir = @"C:\custom\models";
        var (dir, repo) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen17B, modelDir: customDir);
        Assert.Equal(customDir, dir);
        // Repo should still be variant default
        Assert.Contains("1.7B", repo);
    }

    [Fact]
    public void ResolveForVariant_CustomRepoId_OverridesDefault()
    {
        var customRepo = "myorg/custom-model";
        var (dir, repo) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen06B, repoId: customRepo);
        Assert.Equal(customRepo, repo);
        // Dir should still be variant default
        Assert.Equal(ModelDownloader.DefaultModelDir, dir);
    }

    [Fact]
    public void ResolveForVariant_BothOverrides_UsesBoth()
    {
        var customDir = @"C:\my\models";
        var customRepo = "myorg/custom-model";
        var (dir, repo) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen17B, customDir, customRepo);
        Assert.Equal(customDir, dir);
        Assert.Equal(customRepo, repo);
    }

    [Fact]
    public void ResolveForVariant_06BMatchesLegacyPaths()
    {
        // Backward compat: 0.6B resolution must match the legacy defaults exactly
        var (dir, repo) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen06B);
        Assert.Equal(ModelDownloader.DefaultModelDir, dir);
        Assert.Equal(ModelDownloader.DefaultRepoId, repo);
    }

    // ── QwenTtsOptions Variant Integration ──────────────────────────

    [Fact]
    public void QwenTtsOptions_DefaultModelVariant_Is06B()
    {
        var opts = new QwenTtsOptions();
        Assert.Equal(QwenModelVariant.Qwen06B, opts.ModelVariant);
    }

    [Fact]
    public void QwenTtsOptions_CanSet17B()
    {
        var opts = new QwenTtsOptions { ModelVariant = QwenModelVariant.Qwen17B };
        Assert.Equal(QwenModelVariant.Qwen17B, opts.ModelVariant);
    }

    [Fact]
    public void QwenTtsOptions_InstructText_DefaultIsNull()
    {
        var opts = new QwenTtsOptions();
        Assert.Null(opts.InstructText);
    }

    [Fact]
    public void QwenTtsOptions_InstructText_CanBeSet()
    {
        var opts = new QwenTtsOptions { InstructText = "Read with a calm, warm tone" };
        Assert.Equal("Read with a calm, warm tone", opts.InstructText);
    }

    [Fact]
    public void QwenTtsOptions_InstructText_CanBeEmptyString()
    {
        var opts = new QwenTtsOptions { InstructText = "" };
        Assert.Equal("", opts.InstructText);
    }

    // ── TextToSpeechOptions Instruct ────────────────────────────────

    [Fact]
    public void TextToSpeechOptions_Instruct_DefaultIsNull()
    {
        var opts = new TextToSpeechOptions();
        Assert.Null(opts.Instruct);
    }

    [Fact]
    public void TextToSpeechOptions_Instruct_CanBeSet()
    {
        var opts = new TextToSpeechOptions { Instruct = "Speak with excitement" };
        Assert.Equal("Speak with excitement", opts.Instruct);
    }

    // ── Enum exhaustiveness ─────────────────────────────────────────

    [Fact]
    public void AllVariants_HaveUniqueEnumValues()
    {
        var variants = QwenModelVariantConfig.GetAllVariants();
        var values = variants.Select(v => (int)v).ToList();
        Assert.Equal(values.Count, values.Distinct().Count());
    }

    [Fact]
    public void AllVariants_HaveUniqueRepoIds()
    {
        var variants = QwenModelVariantConfig.GetAllVariants();
        var repos = variants.Select(QwenModelVariantConfig.GetRepoId).ToList();
        Assert.Equal(repos.Count, repos.Distinct().Count());
    }

    [Fact]
    public void AllVariants_HaveUniqueSubDirs()
    {
        var variants = QwenModelVariantConfig.GetAllVariants();
        var dirs = variants.Select(QwenModelVariantConfig.GetModelSubDir).ToList();
        Assert.Equal(dirs.Count, dirs.Distinct().Count());
    }

    [Fact]
    public void AllVariants_HaveUniqueDefaultModelDirs()
    {
        var variants = QwenModelVariantConfig.GetAllVariants();
        var dirs = variants.Select(QwenModelVariantConfig.GetDefaultModelDir).ToList();
        Assert.Equal(dirs.Count, dirs.Distinct().Count());
    }
}
