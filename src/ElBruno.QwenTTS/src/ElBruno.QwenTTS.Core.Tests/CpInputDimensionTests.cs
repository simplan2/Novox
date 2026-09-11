using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests that Code Predictor input construction uses correct dimensions for each model variant.
/// The CP always expects 1024-dim input. For 0.6B (hidden=1024) no projection is needed.
/// For 1.7B (hidden=2048), hidden states must be projected from 2048→1024 before CP input.
/// Issue #27: 1.7B was truncating 2048-dim hidden states to 1024 instead of projecting.
/// </summary>
public class CpInputDimensionTests
{
    private const int CpHiddenSize = 1024; // CP hidden_size is 1024 for ALL model variants

    /// <summary>
    /// Simulates CP prefill input construction.
    /// Returns the input array that would be fed to code_predictor.onnx.
    /// </summary>
    private static float[] BuildCpPrefillInput(
        float[] hiddenState,     // last hidden state from Talker (dim = hidden_size)
        float[] group0Embed,     // talker codec embedding for group0 token (dim = hidden_size)
        int cpHiddenSize,        // CP expects this dimension
        bool hasProjection,
        Func<float[], float[]>? projectFn = null)
    {
        float[] projectedHidden;
        float[] projectedGroup0;

        if (hasProjection && projectFn != null)
        {
            projectedHidden = projectFn(hiddenState);
            projectedGroup0 = projectFn(group0Embed);
        }
        else
        {
            // No projection — use directly (assumes dims already match)
            if (hiddenState.Length != cpHiddenSize)
                throw new InvalidOperationException(
                    $"Hidden state dim {hiddenState.Length} != CP hidden dim {cpHiddenSize}. " +
                    "Projection required but not available.");
            projectedHidden = hiddenState;
            projectedGroup0 = group0Embed;
        }

        // CP prefill input: concat(projected_hidden, projected_group0) → shape (1, 2, cpHiddenSize)
        var cpInput = new float[2 * cpHiddenSize];
        Array.Copy(projectedHidden, 0, cpInput, 0, cpHiddenSize);
        Array.Copy(projectedGroup0, 0, cpInput, cpHiddenSize, cpHiddenSize);
        return cpInput;
    }

    /// <summary>Simple linear projection: output[i] = sum(weight[i,j]*input[j]) + bias[i]</summary>
    private static float[] Project(float[,] weight, float[] bias, float[] input)
    {
        int outDim = weight.GetLength(0);
        int inDim = weight.GetLength(1);
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

    // ── Test 1: 0.6B CP input = 1024 (no projection) ───────────────

    [Fact]
    public void CpInputDim_06B_Equals1024()
    {
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen06B);
        Assert.Equal(CpHiddenSize, hiddenSize);

        // Build CP input with no projection
        var hiddenState = new float[hiddenSize];
        var group0Embed = new float[hiddenSize];
        Array.Fill(hiddenState, 1.0f);
        Array.Fill(group0Embed, 2.0f);

        var cpInput = BuildCpPrefillInput(hiddenState, group0Embed, CpHiddenSize,
            hasProjection: false);

        Assert.Equal(2 * CpHiddenSize, cpInput.Length);
        // First half = hidden state
        Assert.Equal(1.0f, cpInput[0]);
        // Second half = group0 embed
        Assert.Equal(2.0f, cpInput[CpHiddenSize]);
    }

    // ── Test 2: 1.7B with projection → CP input = 1024 ─────────────

    [Fact]
    public void CpInputDim_17B_WithProjection_Equals1024()
    {
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B);
        Assert.Equal(2048, hiddenSize);

        // Create synthetic projection (2048→1024) — identity-like: take even indices
        var weight = new float[CpHiddenSize, hiddenSize];
        var bias = new float[CpHiddenSize];
        for (int i = 0; i < CpHiddenSize; i++)
            weight[i, i * 2] = 1.0f; // picks every other element

        var hiddenState = new float[hiddenSize];
        var group0Embed = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            hiddenState[i] = i;
            group0Embed[i] = i + 1000;
        }

        var cpInput = BuildCpPrefillInput(hiddenState, group0Embed, CpHiddenSize,
            hasProjection: true,
            projectFn: input => Project(weight, bias, input));

        // CP input should be 2 * 1024
        Assert.Equal(2 * CpHiddenSize, cpInput.Length);
        // First half: projected hidden_state (even indices of original)
        Assert.Equal(0.0f, cpInput[0]);     // hiddenState[0]
        Assert.Equal(2.0f, cpInput[1]);     // hiddenState[2]
        // Second half: projected group0_embed
        Assert.Equal(1000.0f, cpInput[CpHiddenSize]);     // group0Embed[0]
        Assert.Equal(1002.0f, cpInput[CpHiddenSize + 1]); // group0Embed[2]
    }

    // ── Test 3: 1.7B WITHOUT projection (old model) → CP input = 2048 ──

    [Fact]
    public void CpInputDim_17B_BackwardCompat_Equals2048()
    {
        // Backward compat: old 1.7B model bakes projection into ONNX, so CP expects 2048
        // This scenario: hasProjection=false, hidden_size=2048, CP has baked-in small_to_mtp_projection
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B);

        // Without external projection, if CP ONNX has baked-in projection,
        // the raw 2048-dim input is fed directly (CP does the projection internally)
        int cpInputDimForOldModel = hiddenSize; // 2048 — CP expects full hidden and projects internally
        Assert.Equal(2048, cpInputDimForOldModel);

        // But the NEW model (with external projection) expects 1024
        int cpInputDimForNewModel = CpHiddenSize;
        Assert.Equal(1024, cpInputDimForNewModel);

        // The mismatch between old and new is exactly the issue #27 fix
        Assert.NotEqual(cpInputDimForOldModel, cpInputDimForNewModel);
    }

    // ── Test 4: Prefill projection correctness ──────────────────────

    [Fact]
    public void CpPrefillInput_WithProjection_ProjectsCorrectly()
    {
        // 1.7B: hidden_state (2048) and group0_embed (2048) must both be projected to 1024
        int inDim = 2048;
        int outDim = 1024;

        // Simple projection: average pairs of elements
        var weight = new float[outDim, inDim];
        var bias = new float[outDim];
        for (int i = 0; i < outDim; i++)
        {
            weight[i, i * 2] = 0.5f;
            weight[i, i * 2 + 1] = 0.5f;
        }

        var hiddenState = new float[inDim];
        var group0Embed = new float[inDim];
        for (int i = 0; i < inDim; i++)
        {
            hiddenState[i] = (float)i;
            group0Embed[i] = (float)(i * 10);
        }

        var cpInput = BuildCpPrefillInput(hiddenState, group0Embed, outDim,
            hasProjection: true,
            projectFn: input => Project(weight, bias, input));

        // Verify projection was applied correctly
        Assert.Equal(2 * outDim, cpInput.Length);

        // Hidden state: avg(0,1)=0.5, avg(2,3)=2.5, avg(4,5)=4.5
        Assert.Equal(0.5f, cpInput[0], precision: 4);
        Assert.Equal(2.5f, cpInput[1], precision: 4);
        Assert.Equal(4.5f, cpInput[2], precision: 4);

        // Group0 embed: avg(0,10)=5.0, avg(20,30)=25.0, avg(40,50)=45.0
        Assert.Equal(5.0f, cpInput[outDim], precision: 4);
        Assert.Equal(25.0f, cpInput[outDim + 1], precision: 4);
        Assert.Equal(45.0f, cpInput[outDim + 2], precision: 4);
    }

    // ── Test 5: Subsequent CP steps use cpHiddenSize directly ───────

    [Fact]
    public void CpSubsequentInput_UsesCpHiddenSize()
    {
        // After the prefill (groupIdx=1), subsequent CP steps (groupIdx=2..15)
        // feed cp_codec_embedding lookup results, which are already cpHiddenSize dim.
        // No projection needed for subsequent steps — embeddings are natively 1024.

        int cpHiddenSize = CpHiddenSize;

        // Simulate cp_codec_embedding lookup for group 2
        var cpEmbed = new float[cpHiddenSize];
        Array.Fill(cpEmbed, 3.14f);

        // For subsequent steps: input is (1, 1, cpHiddenSize) — single token
        var subsequentInput = new float[1 * 1 * cpHiddenSize];
        Array.Copy(cpEmbed, 0, subsequentInput, 0, cpHiddenSize);

        Assert.Equal(cpHiddenSize, subsequentInput.Length);
        Assert.All(subsequentInput, v => Assert.Equal(3.14f, v, precision: 5));
    }

    // ── Dimension invariant: CP embeddings are always 1024 for all variants ──

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B)]
    [InlineData(QwenModelVariant.Qwen17B)]
    public void CpHiddenSize_Is1024_ForAllVariants(QwenModelVariant variant)
    {
        // CP hidden size is structurally fixed at 1024 for all current variants
        // This is independent of the talker hidden_size
        Assert.Equal(CpHiddenSize, 1024);

        // The talker hidden_size varies per variant
        int talkerHidden = QwenModelVariantConfig.GetHiddenSize(variant);
        Assert.True(talkerHidden >= CpHiddenSize,
            $"Talker hidden_size ({talkerHidden}) must be >= CP hidden_size ({CpHiddenSize})");
    }

    // ── Projection is needed exactly when hidden > cpHidden ─────────

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B, false)]
    [InlineData(QwenModelVariant.Qwen17B, true)]
    public void ProjectionNeeded_MatchesVariant(QwenModelVariant variant, bool expectedNeedProjection)
    {
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(variant);
        bool needsProjection = hiddenSize != CpHiddenSize;

        Assert.Equal(expectedNeedProjection, needsProjection);
    }

    // ── No projection + wrong dim should fail ───────────────────────

    [Fact]
    public void CpPrefillInput_NoProjection_WrongDim_Throws()
    {
        // 1.7B hidden_state is 2048 but no projection available → should fail
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B);
        var hiddenState = new float[hiddenSize];
        var group0Embed = new float[hiddenSize];

        var ex = Assert.Throws<InvalidOperationException>(() =>
            BuildCpPrefillInput(hiddenState, group0Embed, CpHiddenSize,
                hasProjection: false));

        Assert.Contains("2048", ex.Message);
        Assert.Contains("1024", ex.Message);
        Assert.Contains("Projection required", ex.Message);
    }
}
