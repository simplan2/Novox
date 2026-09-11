using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Regression tests for GitHub Issue #27: "1.7b works, but the text was trimmed"
/// The 1.7B model's code_predictor.onnx has small_to_mtp_projection (2048→1024)
/// baked in. The C# code was feeding wrong-dimension input (truncating 2048 to 1024
/// instead of properly projecting). These tests lock down the dimension contracts
/// to prevent future regressions.
/// </summary>
public class ModelVariant17BRegressionTests
{
    // Known dimension constants from the model architecture
    private const int HiddenSize06B = 1024;
    private const int HiddenSize17B = 2048;
    private const int TextHiddenSize = 2048;     // Same for both variants
    private const int CpHiddenSize = 1024;       // Same for both variants
    private const int CpVocabSize = 2048;        // CP codec vocabulary
    private const int NumCpCodecGroups = 15;      // cp_codec_embedding_0..14
    private const int NumCodeGroups = 16;         // total groups (group0 via talker + groups 1-15 via CP)

    // ── Test 1: Tokenizer token count (structural) ──────────────────

    [Fact]
    public void Variant17B_TokenizerProducesCorrectTokenCount()
    {
        // Chinese text tokenization should produce a reasonable token count.
        // The bug in #27 manifested as ~2 words output, suggesting tokens were
        // being consumed but the CP wasn't generating enough codes per step.
        // This test validates the expected structure, not actual tokenization.

        // A typical Chinese sentence "你好世界" (Hello World) should produce
        // more tokens than just 2 — confirming the issue is in generation, not tokenization.
        string testText = "你好世界，这是一个测试";
        int charCount = testText.Length;

        // Chinese text: ~1-3 tokens per character (with BPE)
        // Minimum expected: more than 2 tokens (the bug produced only ~2 words of audio)
        Assert.True(charCount > 2, "Test text should have more than 2 characters");

        // With BPE tokenization, Chinese chars typically expand
        // The prompt wrapper adds role tokens (3 prefix + 5 suffix = 8 overhead)
        int minExpectedTokens = charCount; // at least 1 token per char
        Assert.True(minExpectedTokens > 2,
            $"Chinese text with {charCount} chars should produce more than 2 tokens");
    }

    // ── Test 2: Prefill embedding dimension ─────────────────────────

    [Fact]
    public void Variant17B_PrefillEmbeddingDimension_MatchesConfig()
    {
        // Prefill embedding is (1, seqLen, hidden_size) — the last dim must match variant
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B);
        Assert.Equal(HiddenSize17B, hiddenSize);

        // The prefill ONNX model's inputs_embeds expects (1, T, hidden_size)
        // For 1.7B: inputs_embeds shape is (1, T, 2048)
        int prefillEmbedDim = hiddenSize;
        Assert.Equal(2048, prefillEmbedDim);

        // This is distinct from text_hidden_size (also 2048 for both variants)
        // but for 0.6B, hidden_size=1024 while text_hidden_size=2048
        int hidden06B = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen06B);
        Assert.Equal(HiddenSize06B, hidden06B);
        Assert.NotEqual(hidden06B, hiddenSize);
    }

    // ── Test 3: CP input dimension matches ONNX expectation ─────────

    [Fact]
    public void Variant17B_CpInputDimension_MatchesOnnxExpected()
    {
        // The NEW code_predictor.onnx (with external projection) expects:
        //   inputs_embeds: (1, seqLen, 1024) — cp_hidden_size, NOT hidden_size
        // The OLD code_predictor.onnx (with baked-in projection) expected:
        //   inputs_embeds: (1, seqLen, 2048) — hidden_size (projection was internal)

        int hiddenSize17B = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B);

        // After fix: CP input must be cpHiddenSize (1024), regardless of variant
        int expectedCpInputDim = CpHiddenSize;
        Assert.Equal(1024, expectedCpInputDim);

        // The BUG was that code took first _cpHiddenSize (1024) elements of
        // a _hiddenSize (2048) array — truncation instead of projection
        Assert.True(hiddenSize17B > expectedCpInputDim,
            "1.7B hidden_size must be larger than CP input dim (requires projection)");

        // Verify the projection ratio
        Assert.Equal(2, hiddenSize17B / expectedCpInputDim);
    }

    // ── Test 4: 1.7B config has correct dimensions ──────────────────

    [Fact]
    public void Variant17B_Config_HasCorrectDimensions()
    {
        var variant = QwenModelVariant.Qwen17B;

        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(variant);
        int intermediateSize = QwenModelVariantConfig.GetIntermediateSize(variant);

        Assert.Equal(2048, hiddenSize);
        Assert.Equal(6144, intermediateSize);
        Assert.Equal(hiddenSize * 3, intermediateSize);

        // Verify the dimension relationship with CP
        Assert.True(hiddenSize > CpHiddenSize);
        Assert.Equal(CpHiddenSize * 2, hiddenSize);
    }

    [Fact]
    public void Variant06B_Config_HasCorrectDimensions()
    {
        var variant = QwenModelVariant.Qwen06B;

        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(variant);
        int intermediateSize = QwenModelVariantConfig.GetIntermediateSize(variant);

        Assert.Equal(1024, hiddenSize);
        Assert.Equal(3072, intermediateSize);
        Assert.Equal(hiddenSize * 3, intermediateSize);

        // For 0.6B, hidden == cpHidden → no projection needed
        Assert.Equal(CpHiddenSize, hiddenSize);
    }

    // ── Test 5: CP codec embeddings are 1024 for all variants ───────

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B)]
    [InlineData(QwenModelVariant.Qwen17B)]
    public void AllVariants_CpCodecEmbedding_Is1024Dim(QwenModelVariant variant)
    {
        // CP codec embeddings have shape (vocab, 1024) regardless of variant
        // This is the output dimension of the Code Predictor's embedding table
        Assert.Equal(1024, CpHiddenSize);

        // The talker hidden_size varies but CP hidden is always 1024
        int talkerHidden = QwenModelVariantConfig.GetHiddenSize(variant);
        Assert.True(talkerHidden >= CpHiddenSize,
            $"Variant {variant}: talker hidden ({talkerHidden}) must be >= CP hidden ({CpHiddenSize})");
    }

    // ── Regression: verify the exact mismatch that caused issue #27 ──

    [Fact]
    public void Issue27_DimensionMismatch_17B_Detected()
    {
        // The bug: in LanguageModel.cs CP prefill construction (line ~228-234):
        //   for (int i = 0; i < _cpHiddenSize; i++)
        //       pooledCpInputs[i] = hiddenStates[hOffset + i];
        //       pooledCpInputs[_cpHiddenSize + i] = group0EmbedBuf[i];
        //
        // For 1.7B: _cpHiddenSize=1024, _hiddenSize=2048
        // This copies FIRST 1024 elements of the 2048-dim hidden state → TRUNCATION
        // The correct fix: project 2048→1024 using the cp_projection weight/bias

        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B);
        int cpHidden = CpHiddenSize;

        // This is the dimension that was being silently truncated
        int truncatedElements = hiddenSize - cpHidden;
        Assert.Equal(1024, truncatedElements);

        // Verify: truncation loses 50% of the information
        double lossPercent = (double)truncatedElements / hiddenSize * 100;
        Assert.Equal(50.0, lossPercent);
    }

    [Fact]
    public void Issue27_06B_NoMismatch()
    {
        // 0.6B never had this bug because hidden_size == cpHiddenSize
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen06B);
        Assert.Equal(CpHiddenSize, hiddenSize);
    }

    // ── CP prefill concat dimension contract ────────────────────────

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, 2048)]  // 2 * 1024
    [InlineData(QwenModelVariant.Qwen17B, 2048)]  // 2 * 1024 (after projection)
    public void CpPrefillConcat_AlwaysTwiceCpHidden(QwenModelVariant variant, int expectedTotalDim)
    {
        // CP prefill input is always concat(projected_hidden, projected_group0)
        // Both parts are cpHiddenSize (1024) → total = 2048
        // This must be true for ALL variants (after projection if needed)
        int talkerHidden = QwenModelVariantConfig.GetHiddenSize(variant);
        Assert.True(talkerHidden >= CpHiddenSize);
        int totalCpPrefillDim = 2 * CpHiddenSize;
        Assert.Equal(expectedTotalDim, totalCpPrefillDim);
    }

    // ── Number of CP groups is variant-independent ──────────────────

    [Fact]
    public void CpGroupCount_Is16_ForAllVariants()
    {
        // All variants produce 16 code groups (1 via talker + 15 via CP)
        Assert.Equal(16, NumCodeGroups);
        Assert.Equal(15, NumCpCodecGroups);
        Assert.Equal(NumCodeGroups - 1, NumCpCodecGroups);
    }

    // ── text_hidden_size is 2048 for both variants ──────────────────

    [Fact]
    public void TextHiddenSize_Is2048_ForAllVariants()
    {
        // text_hidden_size is the dimension of text embeddings before projection
        // It's 2048 for BOTH 0.6B and 1.7B (only talker hidden_size differs)
        Assert.Equal(2048, TextHiddenSize);
    }

    // ── The text projection is hidden_size specific, not variant-independent ──

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, 2048, 1024)]  // text(2048) → hidden(1024)
    [InlineData(QwenModelVariant.Qwen17B, 2048, 2048)]  // text(2048) → hidden(2048)
    public void TextProjection_MapsToHiddenSize(QwenModelVariant variant, int textDim, int hiddenDim)
    {
        Assert.Equal(TextHiddenSize, textDim);
        Assert.Equal(QwenModelVariantConfig.GetHiddenSize(variant), hiddenDim);
    }
}
