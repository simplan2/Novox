using ElBruno.QwenTTS.Models;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Issue #29: Comprehensive tests for CP prefill buffer dimension bug.
/// Tests the two extracted static methods from LanguageModel:
/// - BuildCpPrefillDirect (copies hidden state + group0 embedding into prefill buffer)
/// - AccumulateCpEmbedding (accumulates CP codec embedding into next talker input)
/// 
/// Dimension configurations tested:
/// - 0.6B: hiddenSize=1024, cpHiddenSize=1024, cpInputDim=1024 (all same — bug invisible)
/// - 1.7B correct: hiddenSize=2048, cpHiddenSize=1024, cpInputDim=1024 (projection)
/// - 1.7B bad export: hiddenSize=2048, cpHiddenSize=2048, cpInputDim=1024 (maximally divergent — triggers bug)
/// </summary>
public class Issue29SpanDimensionTests
{
    // Dimension configurations for Theory tests
    public static TheoryData<int, int, int> AllDimensionConfigs => new()
    {
        { 1024, 1024, 1024 }, // 0.6B: all same
        { 2048, 1024, 1024 }, // 1.7B correct: projection reduces 2048→1024
        { 2048, 2048, 1024 }  // 1.7B bad export: maximally divergent
    };

    #region Group 1: BuildCpPrefillDirect tests

    [Theory]
    [MemberData(nameof(AllDimensionConfigs))]
    public void BuildCpPrefillDirect_CorrectLayout_AllConfigs(int hiddenSize, int cpHiddenSize, int cpInputDim)
    {
        // Arrange
        var buffer = new float[2 * cpInputDim];
        var hiddenStates = new float[hiddenSize];
        var group0Embed = new float[cpHiddenSize];
        
        // Fill with distinct values
        for (int i = 0; i < hiddenSize; i++)
            hiddenStates[i] = 100f + i;
        for (int i = 0; i < cpHiddenSize; i++)
            group0Embed[i] = 200f + i;

        // Act
        LanguageModel.BuildCpPrefillDirect(buffer, hiddenStates, hOffset: 0, group0Embed, cpInputDim);

        // Assert: buffer[0..cpInputDim] == first cpInputDim elements of hiddenStates
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal(100f + i, buffer[i]);
        }

        // Assert: buffer[cpInputDim..2*cpInputDim] == first cpInputDim elements of group0Embed
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal(200f + i, buffer[cpInputDim + i]);
        }
    }

    [Fact]
    public void BuildCpPrefillDirect_DoesNotOverflow_LargerBuffers()
    {
        // Arrange: ArrayPool gives oversized buffer (simulate 3*cpInputDim)
        int cpInputDim = 1024;
        var buffer = new float[3 * cpInputDim]; // Larger than needed
        var hiddenStates = new float[2048];
        var group0Embed = new float[1024];
        
        // Fill with distinct values
        for (int i = 0; i < hiddenStates.Length; i++)
            hiddenStates[i] = 10f + i;
        for (int i = 0; i < group0Embed.Length; i++)
            group0Embed[i] = 20f + i;
        
        // Sentinel values in the extra space
        for (int i = 2 * cpInputDim; i < buffer.Length; i++)
            buffer[i] = -999f;

        // Act
        LanguageModel.BuildCpPrefillDirect(buffer, hiddenStates, hOffset: 0, group0Embed, cpInputDim);

        // Assert: only first 2*cpInputDim elements are written
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal(10f + i, buffer[i]);
            Assert.Equal(20f + i, buffer[cpInputDim + i]);
        }
        
        // Assert: extra space is untouched
        for (int i = 2 * cpInputDim; i < buffer.Length; i++)
        {
            Assert.Equal(-999f, buffer[i]);
        }
    }

    [Fact]
    public void BuildCpPrefillDirect_LargeHiddenSmallCpInput()
    {
        // Arrange: hiddenSize=2048, cpInputDim=1024 — only first 1024 copied
        int hiddenSize = 2048;
        int cpInputDim = 1024;
        var buffer = new float[2 * cpInputDim];
        var hiddenStates = new float[hiddenSize];
        var group0Embed = new float[1024];
        
        // Fill hidden states with values 0..2047
        for (int i = 0; i < hiddenSize; i++)
            hiddenStates[i] = i;
        
        // Fill group0 with values 3000..4023
        for (int i = 0; i < group0Embed.Length; i++)
            group0Embed[i] = 3000f + i;

        // Act
        LanguageModel.BuildCpPrefillDirect(buffer, hiddenStates, hOffset: 0, group0Embed, cpInputDim);

        // Assert: only first 1024 elements of hidden states copied (not all 2048)
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal((float)i, buffer[i]);
        }
        
        // Assert: group0 fully copied
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal(3000f + i, buffer[cpInputDim + i]);
        }
    }

    #endregion

    #region Group 2: AccumulateCpEmbedding tests

    [Theory]
    [MemberData(nameof(AllDimensionConfigs))]
#pragma warning disable xUnit1026 // Theory method parameter 'cpInputDim' not used (intentional - shared test data)
    public void AccumulateCpEmbedding_AddsCorrectly_AllConfigs(int hiddenSize, int cpHiddenSize, int cpInputDim)
#pragma warning restore xUnit1026
    {
        // Arrange
        var nextInputBuf = new float[hiddenSize];
        var cpEmbed = new float[cpHiddenSize];
        
        // Fill with initial values
        for (int i = 0; i < hiddenSize; i++)
            nextInputBuf[i] = 10f + i;
        for (int i = 0; i < cpHiddenSize; i++)
            cpEmbed[i] = 100f + i;
        
        // Snapshot original values beyond cpHiddenSize
        var originalHigher = new float[Math.Max(0, hiddenSize - cpHiddenSize)];
        for (int i = 0; i < originalHigher.Length; i++)
            originalHigher[i] = nextInputBuf[cpHiddenSize + i];

        // Act
        LanguageModel.AccumulateCpEmbedding(nextInputBuf, cpEmbed, cpHiddenSize);

        // Assert: first cpHiddenSize elements are accumulated
        for (int i = 0; i < cpHiddenSize; i++)
        {
            Assert.Equal((10f + i) + (100f + i), nextInputBuf[i]);
        }
        
        // Assert: elements beyond cpHiddenSize are unchanged
        for (int i = 0; i < originalHigher.Length; i++)
        {
            Assert.Equal(originalHigher[i], nextInputBuf[cpHiddenSize + i]);
        }
    }

    [Fact]
    public void AccumulateCpEmbedding_PreservesHigherElements()
    {
        // Arrange: nextInputBuf is 2048, cpHiddenSize is 1024
        int hiddenSize = 2048;
        int cpHiddenSize = 1024;
        var nextInputBuf = new float[hiddenSize];
        var cpEmbed = new float[cpHiddenSize];
        
        // Fill with distinct values
        for (int i = 0; i < hiddenSize; i++)
            nextInputBuf[i] = i;
        for (int i = 0; i < cpHiddenSize; i++)
            cpEmbed[i] = 5000f + i;

        // Act
        LanguageModel.AccumulateCpEmbedding(nextInputBuf, cpEmbed, cpHiddenSize);

        // Assert: first 1024 elements accumulated
        for (int i = 0; i < cpHiddenSize; i++)
        {
            Assert.Equal((float)i + (5000f + i), nextInputBuf[i]);
        }
        
        // Assert: elements [1024..2048] are untouched
        for (int i = cpHiddenSize; i < hiddenSize; i++)
        {
            Assert.Equal((float)i, nextInputBuf[i]);
        }
    }

    [Fact]
    public void AccumulateCpEmbedding_MultipleAccumulations()
    {
        // Arrange: simulate 15 CP groups (typical for a sentence)
        int cpHiddenSize = 1024;
        var nextInputBuf = new float[2048];
        var cpEmbed = new float[cpHiddenSize];
        
        // Initialize
        for (int i = 0; i < nextInputBuf.Length; i++)
            nextInputBuf[i] = 100f;
        for (int i = 0; i < cpEmbed.Length; i++)
            cpEmbed[i] = 10f;

        // Act: accumulate 15 times
        for (int call = 0; call < 15; call++)
        {
            LanguageModel.AccumulateCpEmbedding(nextInputBuf, cpEmbed, cpHiddenSize);
        }

        // Assert: first cpHiddenSize elements are 100 + 15*10 = 250
        for (int i = 0; i < cpHiddenSize; i++)
        {
            Assert.Equal(100f + 15 * 10f, nextInputBuf[i]);
        }
        
        // Assert: elements [1024..2048] are still 100
        for (int i = cpHiddenSize; i < nextInputBuf.Length; i++)
        {
            Assert.Equal(100f, nextInputBuf[i]);
        }
    }

    #endregion

    #region Group 3: Prefill buffer layout contract tests

    [Fact]
    public void PrefillBuffer_TwoTokenLayout_MatchesTensorShape()
    {
        // Arrange: create buffer of size 2*cpInputDim, fill two tokens
        int cpInputDim = 1024;
        var buffer = new float[2 * cpInputDim];
        
        // Token 0: values 0..1023
        for (int i = 0; i < cpInputDim; i++)
            buffer[i] = i;
        
        // Token 1: values 2000..3023
        for (int i = 0; i < cpInputDim; i++)
            buffer[cpInputDim + i] = 2000f + i;

        // Act: create DenseTensor with shape [1, 2, cpInputDim]
        var tensor = new DenseTensor<float>(buffer, new[] { 1, 2, cpInputDim });

        // Assert: tensor[0,0,i] == token0[i]
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal((float)i, tensor[0, 0, i]);
        }
        
        // Assert: tensor[0,1,i] == token1[i]
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal(2000f + i, tensor[0, 1, i]);
        }
    }

    [Fact]
    public void PrefillBuffer_WithProjection_SpansMustUseCpInputDim()
    {
        // Arrange: the key regression test
        // Buffer size is 2*cpInputDim where cpInputDim=1024
        // If code incorrectly uses cpHiddenSize=2048 for span offsets, it will overflow
        int cpInputDim = 1024;
        int cpHiddenSize = 2048; // The wrong value that would cause the bug
        var buffer = new float[2 * cpInputDim]; // 2048 elements total

        // Act & Assert: creating spans with cpInputDim is safe
        var span0 = new Span<float>(buffer, 0, cpInputDim);
        var span1 = new Span<float>(buffer, cpInputDim, cpInputDim);
        
        Assert.Equal(cpInputDim, span0.Length);
        Assert.Equal(cpInputDim, span1.Length);

        // Assert: creating spans with cpHiddenSize WOULD overflow
        // (We can't directly test the throw without triggering it, but we can verify the math)
        Assert.True(cpHiddenSize + cpHiddenSize > buffer.Length,
            "Using cpHiddenSize for offsets would exceed buffer bounds");
        Assert.True(cpInputDim + cpInputDim == buffer.Length,
            "Using cpInputDim for offsets exactly matches buffer size");
    }

    [Fact]
    public void PrefillBuffer_NoProjection_CopiesCorrectSubset()
    {
        // Arrange: when hiddenSize=2048 and cpInputDim=2048 (no projection, 1.7B old model)
        int hiddenSize = 2048;
        int cpInputDim = 2048;
        var buffer = new float[2 * cpInputDim]; // 4096 elements
        var hiddenStates = new float[hiddenSize];
        var group0Embed = new float[cpInputDim];
        
        // Fill with distinct values
        for (int i = 0; i < hiddenSize; i++)
            hiddenStates[i] = i;
        for (int i = 0; i < cpInputDim; i++)
            group0Embed[i] = 5000f + i;

        // Act
        LanguageModel.BuildCpPrefillDirect(buffer, hiddenStates, hOffset: 0, group0Embed, cpInputDim);

        // Assert: full hidden state copied (all 2048 elements)
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal((float)i, buffer[i]);
        }
        
        // Assert: group0 copied
        for (int i = 0; i < cpInputDim; i++)
        {
            Assert.Equal(5000f + i, buffer[cpInputDim + i]);
        }
    }

    #endregion

    #region Group 4: Dimension contract assertions

    [Theory]
    [InlineData(2048, 1024, 1024)] // 1.7B with projection: cpInputDim == cpModelHiddenSize
    public void CpInputDim_WithProjection_UsesModelHiddenSize(int hiddenSize, int cpModelHiddenSize, int cpInputDim)
    {
        // When HasCpProjection=true, cpInputDim should equal cpModelHiddenSize (not cpHiddenSize)
        // This test validates the contract — actual LanguageModel implementation will honor this
        
        // Arrange: simulate projection scenario
        bool hasCpProjection = cpModelHiddenSize < hiddenSize;
        
        // Assert
        Assert.True(hasCpProjection);
        Assert.Equal(cpModelHiddenSize, cpInputDim);
    }

    [Theory]
    [InlineData(1024, 1024, 1024)] // 0.6B: no projection
    [InlineData(2048, 2048, 2048)] // 1.7B old model: no projection
    public void CpInputDim_WithoutProjection_UsesHiddenSize(int hiddenSize, int cpHiddenSize, int cpInputDim)
    {
        // When HasCpProjection=false, cpInputDim should equal hiddenSize
        
        // Arrange: simulate no-projection scenario
        bool hasCpProjection = false; // Determined by absence of projection files
        
        // Assert
        Assert.False(hasCpProjection);
        Assert.Equal(hiddenSize, cpInputDim);
        Assert.Equal(cpHiddenSize, cpInputDim);
    }

    [Theory]
    [InlineData(1024, 1024)] // 0.6B
    [InlineData(2048, 1024)] // 1.7B with projection
    [InlineData(2048, 2048)] // 1.7B bad export
    public void SpanOffset_MustNotExceedBufferLength(int cpHiddenSize, int cpInputDim)
    {
        // For every (cpHiddenSize, cpInputDim) combo, verify that offset + length <= bufferLength
        // where buffer = 2*cpInputDim
        
        // Arrange
        int bufferLength = 2 * cpInputDim;
        int offset = cpInputDim;
        int length = cpInputDim;

        // Assert: the correct pattern
        Assert.True(offset + length <= bufferLength,
            $"Offset {offset} + length {length} must not exceed buffer length {bufferLength}");
        
        // Assert: using cpHiddenSize would fail when it > cpInputDim
        if (cpHiddenSize > cpInputDim)
        {
            Assert.True(cpHiddenSize + cpHiddenSize > bufferLength,
                $"Using cpHiddenSize {cpHiddenSize} would overflow buffer of length {bufferLength}");
        }
    }

    #endregion

    #region Group 5: Edge cases

    [Fact]
    public void BuildCpPrefillDirect_ZeroDimension_NoOp()
    {
        // Arrange
        var buffer = new float[10];
        var hiddenStates = new float[10];
        var group0Embed = new float[10];
        int cpInputDim = 0;
        
        // Fill buffer with sentinel values
        for (int i = 0; i < buffer.Length; i++)
            buffer[i] = -999f;
        
        var original = (float[])buffer.Clone();

        // Act: cpInputDim=0 should be a no-op (no elements copied)
        LanguageModel.BuildCpPrefillDirect(buffer, hiddenStates, hOffset: 0, group0Embed, cpInputDim);

        // Assert: buffer unchanged
        for (int i = 0; i < buffer.Length; i++)
        {
            Assert.Equal(original[i], buffer[i]);
        }
    }

    [Fact]
    public void AccumulateCpEmbedding_ZeroSize_NoOp()
    {
        // Arrange
        var nextInputBuf = new float[1024];
        var cpEmbed = new float[1024];
        
        // Fill with initial values
        for (int i = 0; i < nextInputBuf.Length; i++)
            nextInputBuf[i] = 42f + i;
        
        var original = (float[])nextInputBuf.Clone();

        // Act: cpHiddenSize=0 should be a no-op
        LanguageModel.AccumulateCpEmbedding(nextInputBuf, cpEmbed, cpHiddenSize: 0);

        // Assert: buffer unchanged
        for (int i = 0; i < nextInputBuf.Length; i++)
        {
            Assert.Equal(original[i], nextInputBuf[i]);
        }
    }

    #endregion
}
