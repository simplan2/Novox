using ElBruno.QwenTTS.Models;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests for NpyReader SEC-3 file size validation (Issue #25).
/// 
/// The NPY size limit was raised from 500 MB to 2 GB to support the
/// 1.7B model's text_embedding.npy (~1.2 GB).
/// 
/// Uses sparse files (SetLength without writing) so tests run fast
/// without allocating gigabytes of disk space.
/// </summary>
public class NpyReaderSizeLimitTests : IDisposable
{
    private readonly string _tempDir;

    public NpyReaderSizeLimitTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_sizelimit_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    #region Rejection Tests — Files Over 2 GB Limit

    [Fact]
    public void ReadFloat1D_FileOver2GB_ThrowsInvalidOperationException()
    {
        var path = CreateSparseFile("oversized_f1d.npy", 2_500_000_000L);

        var ex = Assert.Throws<InvalidOperationException>(() => NpyReader.ReadFloat1D(path));
        Assert.Contains("too large", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReadFloat2D_FileOver2GB_ThrowsInvalidOperationException()
    {
        var path = CreateSparseFile("oversized_f2d.npy", 2_500_000_000L);

        var ex = Assert.Throws<InvalidOperationException>(() => NpyReader.ReadFloat2D(path));
        Assert.Contains("too large", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReadInt64_1D_FileOver2GB_ThrowsInvalidOperationException()
    {
        var path = CreateSparseFile("oversized_i64.npy", 2_500_000_000L);

        var ex = Assert.Throws<InvalidOperationException>(() => NpyReader.ReadInt64_1D(path));
        Assert.Contains("too large", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReadFloat1D_FileJustOver2GB_ThrowsInvalidOperationException()
    {
        var path = CreateSparseFile("just_over.npy", 2_000_000_001L);

        Assert.Throws<InvalidOperationException>(() => NpyReader.ReadFloat1D(path));
    }

    #endregion

    #region Error Message Format Tests

    [Fact]
    public void ErrorMessage_ContainsActualFileSize()
    {
        var path = CreateSparseFile("msg_size.npy", 3_000_000_000L);

        var ex = Assert.Throws<InvalidOperationException>(() => NpyReader.ReadFloat1D(path));

        // Message should include the actual file size in MB (3000.00 MB)
        Assert.Contains("3000.00", ex.Message);
    }

    [Fact]
    public void ErrorMessage_ContainsMaximumAllowedSize()
    {
        var path = CreateSparseFile("msg_max.npy", 3_000_000_000L);

        var ex = Assert.Throws<InvalidOperationException>(() => NpyReader.ReadFloat1D(path));

        // Message should mention the 2 GB maximum (2000.00 MB)
        Assert.Contains("2000.00", ex.Message);
    }

    #endregion

    #region Size Check Precedes Content Validation

    [Fact]
    public void SizeCheck_HappensBeforeContentValidation()
    {
        // An oversized file with non-NPY content should fail on SIZE, not on magic bytes.
        // This proves the size check is the first guard.
        var path = CreateSparseFile("not_npy_but_huge.npy", 2_500_000_000L);

        var ex = Assert.Throws<InvalidOperationException>(() => NpyReader.ReadFloat1D(path));
        Assert.Contains("too large", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    #endregion

    #region Acceptance Tests — Files Within 2 GB Limit

    [Fact]
    public void ReadFloat1D_SmallValidFile_Succeeds()
    {
        // A valid NPY file well under the limit must still work.
        var path = Path.Combine(_tempDir, "small_valid.npy");
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        WriteValidNpyFloat1D(path, data);

        var result = NpyReader.ReadFloat1D(path);

        Assert.Equal(data.Length, result.Length);
        Assert.Equal(1.0f, result[0]);
        Assert.Equal(3.0f, result[2]);
    }

    [Fact]
    public void ReadFloat1D_FileExactlyAt2GB_DoesNotThrowSizeError()
    {
        // Exactly at the boundary (2,000,000,000 bytes) should pass the size check.
        // It will fail on content validation (not a valid NPY), but NOT on size.
        var path = CreateSparseFile("exactly_2gb.npy", 2_000_000_000L);

        // Should throw InvalidDataException (bad magic), NOT InvalidOperationException (too large)
        var ex = Assert.ThrowsAny<Exception>(() => NpyReader.ReadFloat1D(path));
        Assert.IsNotType<InvalidOperationException>(ex);
    }

    #endregion

    #region 1.7B Model Scenario

    [Fact]
    public void NpyLimit_Allows1_2GB_TextEmbedding()
    {
        // The 1.7B model's text_embedding.npy is ~1.2 GB.
        // Size check must pass. Content validation will fail (sparse file), but that's fine.
        var path = CreateSparseFile("text_embedding_17b.npy", 1_200_000_000L);

        // Should NOT throw InvalidOperationException (size rejection)
        var ex = Assert.ThrowsAny<Exception>(() => NpyReader.ReadFloat1D(path));
        Assert.IsNotType<InvalidOperationException>(ex);
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Creates a sparse file that reports the given length without allocating disk space.
    /// On NTFS, SetLength creates a sparse file; only metadata is written.
    /// </summary>
    private string CreateSparseFile(string name, long size)
    {
        var path = Path.Combine(_tempDir, name);
        using var fs = File.Create(path);
        fs.SetLength(size);
        return path;
    }

    /// <summary>
    /// Writes a valid NumPy v1.0 float32 1D file for acceptance tests.
    /// </summary>
    private static void WriteValidNpyFloat1D(string path, float[] data)
    {
        var header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({data.Length},), }}";
        var headerBytes = System.Text.Encoding.ASCII.GetBytes(header);
        int preambleLen = 10;
        int totalHeaderLen = preambleLen + headerBytes.Length + 1;
        int padding = (64 - totalHeaderLen % 64) % 64;
        int paddedHeaderLen = headerBytes.Length + padding + 1;

        using var fs = File.Create(path);
        fs.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });
        fs.Write(new byte[] { 1, 0 });
        fs.Write(BitConverter.GetBytes((ushort)paddedHeaderLen));
        fs.Write(headerBytes);
        for (int i = 0; i < padding; i++) fs.WriteByte((byte)' ');
        fs.WriteByte((byte)'\n');
        var buf = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, buf, 0, buf.Length);
        fs.Write(buf);
    }

    #endregion

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }
}
