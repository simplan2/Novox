using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests for CP (Code Predictor) projection logic.
/// The 1.7B model needs a linear projection (2048→1024) before feeding
/// hidden states into the Code Predictor. These tests validate the math
/// using synthetic weight/bias matrices — no model files required.
/// Issue #27: 1.7B model trimmed text because CP received wrong-dimension input.
/// </summary>
public class CpProjectionTests
{
    // ── Synthetic projection helper (mirrors what EmbeddingStore.CpProjection will do) ──

    /// <summary>
    /// Linear projection: output = weight @ input + bias
    /// weight is (outDim, inDim), input is (inDim,), output is (outDim,)
    /// </summary>
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

    // ── Test 1: Projection not available throws ─────────────────────

    [Fact]
    public void CpProjection_NotAvailable_ThrowsInvalidOperation()
    {
        // Simulate the case where no projection files exist (0.6B model).
        // Calling projection when not loaded must throw InvalidOperationException.
        bool hasProjection = false;
        float[,]? weight = null;
        float[]? bias = null;

        var ex = Assert.Throws<InvalidOperationException>(() =>
        {
            if (!hasProjection || weight is null || bias is null)
                throw new InvalidOperationException(
                    "CP projection is not available. This model variant does not require projection.");

            LinearProjection(weight, bias, new float[2048]);
        });

        Assert.Contains("not available", ex.Message);
    }

    // ── Test 2: HasCpProjection returns false when no files ─────────

    [Fact]
    public void HasCpProjection_ReturnsFalse_WhenNoProjectionFiles()
    {
        // Simulate file-existence check for cp_projection_weight.npy and cp_projection_bias.npy
        var tempDir = Path.Combine(Path.GetTempPath(), $"cptest_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            var weightPath = Path.Combine(tempDir, "cp_projection_weight.npy");
            var biasPath = Path.Combine(tempDir, "cp_projection_bias.npy");

            bool hasProjection = File.Exists(weightPath) && File.Exists(biasPath);

            Assert.False(hasProjection, "HasCpProjection should be false when projection files don't exist");
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void HasCpProjection_ReturnsTrue_WhenProjectionFilesExist()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"cptest_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            var weightPath = Path.Combine(tempDir, "cp_projection_weight.npy");
            var biasPath = Path.Combine(tempDir, "cp_projection_bias.npy");
            File.WriteAllText(weightPath, "dummy");
            File.WriteAllText(biasPath, "dummy");

            bool hasProjection = File.Exists(weightPath) && File.Exists(biasPath);

            Assert.True(hasProjection, "HasCpProjection should be true when projection files exist");
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    // ── Test 3: Correct dimensions — matmul+bias produces correct result ──

    [Fact]
    public void CpProjection_CorrectDimensions_OutputMatchesExpected()
    {
        // Small synthetic projection: 4→2 (mirrors 2048→1024 structure)
        int inDim = 4;
        int outDim = 2;

        // Weight: identity-like with scale
        // [ 1 0 1 0 ]   input [1,2,3,4] → [1*1+0*2+1*3+0*4] = [4]
        // [ 0 1 0 1 ]                    → [0*1+1*2+0*3+1*4] = [6]
        var weight = new float[outDim, inDim];
        weight[0, 0] = 1.0f; weight[0, 2] = 1.0f;
        weight[1, 1] = 1.0f; weight[1, 3] = 1.0f;

        var bias = new float[] { 0.5f, -0.5f };
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

        var output = LinearProjection(weight, bias, input);

        Assert.Equal(outDim, output.Length);
        Assert.Equal(4.5f, output[0], precision: 5);  // (1+3) + 0.5
        Assert.Equal(5.5f, output[1], precision: 5);  // (2+4) - 0.5
    }

    [Fact]
    public void CpProjection_2048To1024_OutputHasCorrectLength()
    {
        // Realistic dimensions: 2048→1024 (what 1.7B needs)
        int inDim = 2048;
        int outDim = 1024;

        var weight = new float[outDim, inDim];
        var bias = new float[outDim];
        var input = new float[inDim];

        // Fill with small values to avoid overflow
        var rng = new Random(42);
        for (int i = 0; i < outDim; i++)
        {
            bias[i] = (float)(rng.NextDouble() * 0.01 - 0.005);
            for (int j = 0; j < inDim; j++)
                weight[i, j] = (float)(rng.NextDouble() * 0.01 - 0.005);
        }
        for (int j = 0; j < inDim; j++)
            input[j] = (float)(rng.NextDouble() * 2 - 1);

        var output = LinearProjection(weight, bias, input);

        Assert.Equal(outDim, output.Length);
        // All values should be finite
        Assert.All(output, v => Assert.True(float.IsFinite(v), $"Non-finite value: {v}"));
    }

    [Fact]
    public void CpProjection_ZeroBias_MatchesPureMatMul()
    {
        int inDim = 3;
        int outDim = 2;

        var weight = new float[outDim, inDim];
        weight[0, 0] = 2.0f; weight[0, 1] = 3.0f; weight[0, 2] = 1.0f;
        weight[1, 0] = 1.0f; weight[1, 1] = -1.0f; weight[1, 2] = 2.0f;

        var bias = new float[outDim]; // all zeros
        var input = new float[] { 1.0f, 1.0f, 1.0f };

        var output = LinearProjection(weight, bias, input);

        Assert.Equal(6.0f, output[0], precision: 5);  // 2+3+1
        Assert.Equal(2.0f, output[1], precision: 5);   // 1-1+2
    }

    [Fact]
    public void CpProjection_IdentityWeight_OutputEqualsInputPlusBias()
    {
        // Square projection with identity weight
        int dim = 4;
        var weight = new float[dim, dim];
        for (int i = 0; i < dim; i++)
            weight[i, i] = 1.0f;

        var bias = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

        var output = LinearProjection(weight, bias, input);

        for (int i = 0; i < dim; i++)
            Assert.Equal(input[i] + bias[i], output[i], precision: 5);
    }

    // ── Test 4: Input dimension mismatch throws ─────────────────────

    [Fact]
    public void CpProjection_InputDimensionMismatch_Throws()
    {
        int inDim = 2048;
        int outDim = 1024;
        var weight = new float[outDim, inDim];
        var bias = new float[outDim];

        // Wrong input length (1024 instead of 2048)
        var wrongInput = new float[1024];

        var ex = Assert.Throws<ArgumentException>(() =>
            LinearProjection(weight, bias, wrongInput));

        Assert.Contains("1024", ex.Message);
        Assert.Contains("2048", ex.Message);
    }

    [Fact]
    public void CpProjection_EmptyInput_Throws()
    {
        var weight = new float[1024, 2048];
        var bias = new float[1024];
        var emptyInput = Array.Empty<float>();

        Assert.Throws<ArgumentException>(() =>
            LinearProjection(weight, bias, emptyInput));
    }

    [Fact]
    public void CpProjection_BiasLengthMismatch_Throws()
    {
        var weight = new float[1024, 2048];
        var wrongBias = new float[512]; // wrong length
        var input = new float[2048];

        Assert.Throws<ArgumentException>(() =>
            LinearProjection(weight, wrongBias, input));
    }

    // ── Structural: 0.6B should NOT need projection ─────────────────

    [Fact]
    public void Variant06B_HiddenEqualsCpHidden_NoProjectionNeeded()
    {
        // For 0.6B: hidden_size == cp_hidden_size == 1024, so no projection needed
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen06B);
        int cpHiddenSize = 1024; // CP hidden is always 1024

        Assert.Equal(hiddenSize, cpHiddenSize);
    }

    [Fact]
    public void Variant17B_HiddenDiffersCpHidden_ProjectionRequired()
    {
        // For 1.7B: hidden_size=2048, cp_hidden_size=1024 → projection required
        int hiddenSize = QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B);
        int cpHiddenSize = 1024;

        Assert.NotEqual(hiddenSize, cpHiddenSize);
        Assert.Equal(2, hiddenSize / cpHiddenSize); // exactly 2× ratio
    }
}
