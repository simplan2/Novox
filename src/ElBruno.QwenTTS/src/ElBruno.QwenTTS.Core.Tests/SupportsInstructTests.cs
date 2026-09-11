using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests for SupportsInstruct() — the variant-gating mechanism that controls
/// whether instruction text (emotion, rate, timbre) is honored or ignored.
/// This is the key 1.7B-specific feature.
/// </summary>
public class SupportsInstructTests
{
    // ── Basic SupportsInstruct ──────────────────────────────────────

    [Fact]
    public void SupportsInstruct_06B_ReturnsFalse()
    {
        Assert.False(QwenModelVariantConfig.SupportsInstruct(QwenModelVariant.Qwen06B));
    }

    [Fact]
    public void SupportsInstruct_17B_ReturnsTrue()
    {
        Assert.True(QwenModelVariantConfig.SupportsInstruct(QwenModelVariant.Qwen17B));
    }

    [Fact]
    public void SupportsInstruct_DefaultVariant_ReturnsFalse()
    {
        // Default variant (0.6B) does NOT support instruct
        Assert.False(QwenModelVariantConfig.SupportsInstruct(QwenModelVariantConfig.Default));
    }

    [Fact]
    public void SupportsInstruct_InvalidVariant_ReturnsFalse()
    {
        // SupportsInstruct uses _ => false (not throw) for unknown variants
        Assert.False(QwenModelVariantConfig.SupportsInstruct((QwenModelVariant)99));
    }

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, false)]
    [InlineData(QwenModelVariant.Qwen17B, true)]
    public void SupportsInstruct_AllVariants_MatchExpected(QwenModelVariant variant, bool expected)
    {
        Assert.Equal(expected, QwenModelVariantConfig.SupportsInstruct(variant));
    }

    // ── Only 1.7B+ supports instruct (structural invariant) ────────

    [Fact]
    public void SupportsInstruct_OnlyVariantsWithHiddenSize2048OrMore()
    {
        // If SupportsInstruct is true, hidden_size must be >= 2048
        foreach (var variant in QwenModelVariantConfig.GetAllVariants())
        {
            if (QwenModelVariantConfig.SupportsInstruct(variant))
            {
                Assert.True(QwenModelVariantConfig.GetHiddenSize(variant) >= 2048,
                    $"{variant} supports instruct but has hidden_size < 2048");
            }
        }
    }
}
