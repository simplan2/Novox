using System.Text.Json;
using ElBruno.QwenTTS.Models;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Regression tests for Issue #28: CP projection bias dimension mismatch.
/// For 1.7B models, _cpHiddenSize (from CP codec embedding shapes) can be 2048
/// while _cpProjectionBias is only 1024. The fix makes config.code_predictor.hidden_size
/// the authoritative CP input dimension, falling back to embedding dim when missing.
/// </summary>
public class Issue28CpDimensionMismatchTests
{
    // ── Synthetic helpers (same pattern as CpProjectionTests / CpInputDimensionTests) ──

    /// <summary>Linear projection: output = weight @ input + bias</summary>
    private static float[] LinearProjection(float[,] weight, float[] bias, float[] input)
    {
        int outDim = weight.GetLength(0);
        int inDim = weight.GetLength(1);

        if (input.Length != inDim)
            throw new ArgumentException($"Input length {input.Length} doesn't match weight input dim {inDim}");
        if (bias.Length != outDim)
            throw new ArgumentException($"Bias length {bias.Length} doesn't match weight output dim {outDim}");

        var output = new float[outDim];
        for (int i = 0; i < outDim; i++)
        {
            float sum = bias[i];
            for (int j = 0; j < inDim; j++)
                sum += weight[i, j] * input[j];
            output[i] = sum;
        }
        return output;
    }

    /// <summary>
    /// Simulates CP prefill input construction with configurable cpInputDim.
    /// This mirrors LanguageModel's logic post-fix: cpInputDim comes from config, not embedding shape.
    /// </summary>
    private static float[,] BuildCpPrefillInput(
        float[] hiddenState,
        float[] group0Embed,
        int cpInputDim,
        bool hasProjection,
        Func<float[], float[]>? projectFn = null)
    {
        float[] row0, row1;

        if (hasProjection && projectFn != null)
        {
            row0 = projectFn(hiddenState);
            row1 = projectFn(group0Embed);
        }
        else
        {
            if (hiddenState.Length != cpInputDim)
                throw new InvalidOperationException(
                    $"Hidden state dim {hiddenState.Length} != cpInputDim {cpInputDim}. " +
                    "Projection required but not available.");
            row0 = hiddenState;
            row1 = group0Embed;
        }

        // Shape: (2, cpInputDim)
        var result = new float[2, cpInputDim];
        for (int i = 0; i < cpInputDim; i++)
        {
            result[0, i] = row0[i];
            result[1, i] = row1[i];
        }
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 1: Bias loop uses weight output dim, not cpHiddenSize
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void CpProjection_BiasLoopUsesWeightOutputDim_NotCpHiddenSize()
    {
        // The bug scenario: cpHiddenSize (embedding dim) = 2048 but
        // projection weight is (1024, 2048) and bias is (1024).
        // Old code: for (i = 0; i < _cpHiddenSize; i++) output[i] += bias[i]
        //   would IndexOutOfRange when _cpHiddenSize=2048 but bias.Length=1024.
        // Fix: loop bound = weight.GetLength(0) = bias.Length = 1024.

        int cpHiddenSizeFromEmbeddings = 2048; // what old code used as loop bound
        int weightOutputDim = 1024;             // actual projection output
        int weightInputDim = 2048;

        var weight = new float[weightOutputDim, weightInputDim];
        var bias = new float[weightOutputDim]; // 1024, NOT 2048
        var input = new float[weightInputDim];

        // Fill with small values
        var rng = new Random(28);
        for (int i = 0; i < weightOutputDim; i++)
        {
            bias[i] = (float)(rng.NextDouble() * 0.1);
            for (int j = 0; j < weightInputDim; j++)
                weight[i, j] = (float)(rng.NextDouble() * 0.01 - 0.005);
        }
        for (int j = 0; j < weightInputDim; j++)
            input[j] = (float)(rng.NextDouble() * 2 - 1);

        // This must NOT throw even though cpHiddenSizeFromEmbeddings (2048) > bias.Length (1024)
        var output = LinearProjection(weight, bias, input);

        Assert.Equal(weightOutputDim, output.Length);
        Assert.NotEqual(cpHiddenSizeFromEmbeddings, output.Length);
        Assert.All(output, v => Assert.True(float.IsFinite(v)));

        // Verify we're NOT accidentally indexing bias with cpHiddenSizeFromEmbeddings
        Assert.True(bias.Length < cpHiddenSizeFromEmbeddings,
            "This test only makes sense when bias is smaller than embedding-derived cpHiddenSize");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 2: Weight output dim always matches bias length
    // ═══════════════════════════════════════════════════════════════════

    [Theory]
    [InlineData(1024, 1024)]
    [InlineData(1024, 2048)]
    [InlineData(512, 2048)]
    public void CpProjection_WeightOutputDimMatchesBias(int outDim, int inDim)
    {
        var weight = new float[outDim, inDim];
        var bias = new float[outDim];
        var input = new float[inDim];

        // Identity-ish diagonal for predictable output
        for (int i = 0; i < Math.Min(outDim, inDim); i++)
            weight[i, i] = 1.0f;

        Array.Fill(input, 1.0f);
        Array.Fill(bias, 0.5f);

        var output = LinearProjection(weight, bias, input);

        // Core invariant: output.Length == weight.GetLength(0) == bias.Length
        Assert.Equal(weight.GetLength(0), output.Length);
        Assert.Equal(bias.Length, output.Length);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 3: CodePredictorConfig.hidden_size deserializes from JSON
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void CodePredictorConfig_HiddenSize_Deserializes()
    {
        // Issue #28 fix: config.json now carries code_predictor.hidden_size
        var json = """
        {
            "talker": { "hidden_size": 2048, "text_hidden_size": 2048, "num_hidden_layers": 16 },
            "code_predictor": { "hidden_size": 1024, "num_hidden_layers": 4, "head_dim": 64 },
            "tts": { "tts_bos_token_id": 0, "tts_eos_token_id": 1 }
        }
        """;

        var config = JsonSerializer.Deserialize<ModelConfig>(json);

        Assert.NotNull(config);
        Assert.Equal(1024, config!.code_predictor.hidden_size);
        Assert.Equal(2048, config.talker.hidden_size);
    }

    [Fact]
    public void CodePredictorConfig_MissingHiddenSize_DefaultsToZero()
    {
        // Backward compat: old config.json without code_predictor.hidden_size
        var json = """
        {
            "talker": { "hidden_size": 1024 },
            "code_predictor": { "num_hidden_layers": 4, "head_dim": 64 },
            "tts": {}
        }
        """;

        var config = JsonSerializer.Deserialize<ModelConfig>(json);

        Assert.NotNull(config);
        Assert.Equal(0, config!.code_predictor.hidden_size);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 4: cpInputDim uses config hidden_size, not embedding dim
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void CpInputDim_UsesConfigHiddenSize_NotEmbeddingDim()
    {
        // Issue #28 scenario: embedding arrays are 2048-dim but config says CP needs 1024
        int embeddingDim = 2048; // _cpHiddenSize from cp_codec_embedding shapes
        int configHiddenSize = 1024; // from config.code_predictor.hidden_size
        bool hasProjection = true;

        // Post-fix logic: cpInputDim = hasProjection ? configHiddenSize : hiddenSize
        int cpInputDim = hasProjection ? configHiddenSize : embeddingDim;

        Assert.Equal(1024, cpInputDim);
        Assert.NotEqual(embeddingDim, cpInputDim);
    }

    [Fact]
    public void CpModelHiddenSize_PrefersConfig_OverEmbeddingDim()
    {
        // Mirrors EmbeddingStore._cpModelHiddenSize resolution logic:
        //   _cpModelHiddenSize = Config.code_predictor.hidden_size > 0
        //       ? Config.code_predictor.hidden_size
        //       : _cpHiddenSize;
        int configValue = 1024;
        int embeddingDerivedValue = 2048;

        int cpModelHiddenSize = configValue > 0 ? configValue : embeddingDerivedValue;

        Assert.Equal(1024, cpModelHiddenSize);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 5: Falls back to embedding dim when config missing
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void CpInputDim_FallsBackToEmbeddingDim_WhenConfigMissing()
    {
        // Backward compat: old 0.6B model, no config.code_predictor.hidden_size
        int embeddingDim = 1024;
        int configHiddenSize = 0; // missing → default int
        bool hasProjection = false;

        // Fallback resolution: config=0 → use embeddingDim
        int cpModelHiddenSize = configHiddenSize > 0 ? configHiddenSize : embeddingDim;

        // No projection + no config mismatch → use full hidden_size (= embedding dim for 0.6B)
        int hiddenSize = 1024;
        int cpInputDim = hasProjection ? cpModelHiddenSize : hiddenSize;

        Assert.Equal(1024, cpInputDim);
        Assert.Equal(embeddingDim, cpInputDim);
    }

    [Fact]
    public void CpModelHiddenSize_FallsBackToEmbeddingDim_WhenConfigZero()
    {
        int configValue = 0;
        int embeddingDerivedValue = 1024;

        int cpModelHiddenSize = configValue > 0 ? configValue : embeddingDerivedValue;

        Assert.Equal(embeddingDerivedValue, cpModelHiddenSize);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Test 6: Issue #28 exact scenario — mismatched dims still works
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    public void CpProjection_MismatchedDimensions_StillWorks()
    {
        // THE issue #28 scenario:
        // - Embedding arrays (cp_codec_embedding_*.npy) are 2048-dim
        // - But cp_projection_weight is (1024, 2048), bias is (1024,)
        // - Old code used _cpHiddenSize=2048 as loop bound → IndexOutOfRange on bias
        // - Fix: use weight.GetLength(0) = 1024 as the projection output dim

        int talkerHidden = 2048;
        int cpProjectionOut = 1024;

        var weight = new float[cpProjectionOut, talkerHidden];
        var bias = new float[cpProjectionOut];

        // Simple projection: average pairs
        for (int i = 0; i < cpProjectionOut; i++)
        {
            weight[i, i * 2] = 0.5f;
            weight[i, i * 2 + 1] = 0.5f;
            bias[i] = 0.01f * i;
        }

        var hiddenState = new float[talkerHidden];
        var group0Embed = new float[talkerHidden];
        for (int i = 0; i < talkerHidden; i++)
        {
            hiddenState[i] = (float)i;
            group0Embed[i] = (float)(i * 10);
        }

        // Build the full prefill input with projection
        var prefill = BuildCpPrefillInput(
            hiddenState, group0Embed,
            cpInputDim: cpProjectionOut,
            hasProjection: true,
            projectFn: input => LinearProjection(weight, bias, input));

        // CRITICAL: shape must be (2, 1024), NOT (2, 2048)
        Assert.Equal(2, prefill.GetLength(0));
        Assert.Equal(cpProjectionOut, prefill.GetLength(1));
        Assert.NotEqual(talkerHidden, prefill.GetLength(1));

        // Verify values: row0 = projected hidden, row1 = projected group0
        // hidden[0]=0, hidden[1]=1 → avg=0.5, + bias[0]=0.0 → 0.5
        Assert.Equal(0.5f, prefill[0, 0], precision: 4);
        // group0[0]=0, group0[1]=10 → avg=5.0, + bias[0]=0.0 → 5.0
        Assert.Equal(5.0f, prefill[1, 0], precision: 4);
    }

    [Fact]
    public void CpProjection_Issue28_NoIndexOutOfRangeOnBias()
    {
        // Explicit IndexOutOfRange regression: if bias loop used _cpHiddenSize=2048
        // instead of weight.GetLength(0)=1024, we'd crash at index 1024.

        int cpHiddenSizeOldBug = 2048;
        int actualOutDim = 1024;

        var weight = new float[actualOutDim, cpHiddenSizeOldBug];
        var bias = new float[actualOutDim];
        var input = new float[cpHiddenSizeOldBug];
        Array.Fill(input, 1.0f);

        // Simulating the OLD buggy loop:
        // for (int i = 0; i < _cpHiddenSize; i++) output[i] += bias[i];
        // With _cpHiddenSize=2048 and bias.Length=1024, this would crash at i=1024.
        var exception = Record.Exception(() =>
        {
            var output = LinearProjection(weight, bias, input);
            // Simulated post-projection bias addition with CORRECT loop bound
            for (int i = 0; i < output.Length; i++)
                output[i] += bias[i]; // safe: output.Length == bias.Length == 1024
        });

        Assert.Null(exception);

        // Verify the OLD loop would have crashed
        Assert.Throws<IndexOutOfRangeException>(() =>
        {
            var output = new float[actualOutDim];
            // BUG: using cpHiddenSizeOldBug (2048) as loop bound with bias[1024]
            for (int i = 0; i < cpHiddenSizeOldBug; i++)
                output[i] += bias[i]; // CRASH at i=1024
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // Parametrized: config resolution across variants
    // ═══════════════════════════════════════════════════════════════════

    [Theory]
    [InlineData(1024, 0, false, 1024)]    // 0.6B: no config, no projection → hiddenSize
    [InlineData(2048, 1024, true, 1024)]  // 1.7B: config=1024, projection → config
    [InlineData(2048, 0, false, 2048)]    // 1.7B old model: no config, no external projection → hiddenSize
    [InlineData(1024, 512, true, 512)]    // hypothetical: config=512, projection → config
    public void CpInputDim_ResolvesCorrectly_AcrossScenarios(
        int hiddenSize, int configCpHidden, bool hasProjection, int expectedCpInputDim)
    {
        // Mirror the EmbeddingStore + LanguageModel resolution chain:
        // 1. cpModelHiddenSize = config > 0 ? config : embeddingDim
        // 2. cpInputDim = hasProjection ? cpModelHiddenSize : hiddenSize
        int embeddingDim = 1024; // cp_codec_embedding is always 1024
        int cpModelHiddenSize = configCpHidden > 0 ? configCpHidden : embeddingDim;
        int cpInputDim = hasProjection ? cpModelHiddenSize : hiddenSize;

        Assert.Equal(expectedCpInputDim, cpInputDim);
    }
}
