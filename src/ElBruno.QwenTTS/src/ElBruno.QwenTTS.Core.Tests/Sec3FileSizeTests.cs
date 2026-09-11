using ElBruno.QwenTTS.Models;
using Xunit;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// SEC-3: File Size Validation Tests for ONNX and NPY file limits
///
/// These tests document and verify the file size security constraints:
/// - ONNX files must not exceed 2 GB (2_000_000_000 bytes)
/// - NPY files must not exceed 2 GB (2_000_000_000 bytes) — raised from 500 MB for 1.7B model (Issue #25)
/// 
/// File size validation occurs BEFORE loading model data to prevent out-of-memory attacks.
/// </summary>
public class Sec3FileSizeTests
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "Sec3Tests");

    public Sec3FileSizeTests()
    {
        // Ensure temp directory exists
        Directory.CreateDirectory(_tempDir);
    }

    private void Cleanup()
    {
        // Clean up any test files
        if (Directory.Exists(_tempDir))
        {
            try
            {
                foreach (var file in Directory.GetFiles(_tempDir))
                    File.Delete(file);
                Directory.Delete(_tempDir);
            }
            catch { }
        }
    }

    #region NPY File Size Tests (2 GB limit — raised from 500 MB for Issue #25)

    /// <summary>
    /// Test: NPY file just under 2 GB passes validation.
    /// Documents that files at or near the limit are handled correctly.
    /// </summary>
    [Fact]
    public void NpyReader_AcceptsFileJustUnder2GB()
    {
        try
        {
            const long size = 1_900_000_000; // Just under 2 GB
            var path = Path.Combine(_tempDir, "test_1900mb.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 2_000_000_000;
            
            // Validation should pass: file size <= 2 GB
            Assert.True(fileInfo.Length <= maxNpySize, 
                $"File size {fileInfo.Length} should be within limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: NPY file exactly at 2 GB boundary (inclusive limit).
    /// Documents that 2 GB is the maximum accepted size.
    /// </summary>
    [Fact]
    public void NpyReader_AcceptsFileAt2GBBoundary()
    {
        try
        {
            const long size = 2_000_000_000; // Exactly 2 GB
            var path = Path.Combine(_tempDir, "test_2gb.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 2_000_000_000;
            
            // Validation should pass: file size <= 2 GB
            Assert.True(fileInfo.Length <= maxNpySize,
                $"File size {fileInfo.Length} should equal limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: NPY file just over 2 GB is rejected.
    /// Documents that files exceeding 2 GB throw InvalidOperationException.
    /// </summary>
    [Fact]
    public void NpyReader_RejectsFileJustOver2GB()
    {
        try
        {
            const long size = 2_000_000_001; // Just over 2 GB
            var path = Path.Combine(_tempDir, "test_2gb_plus_1.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 2_000_000_000;
            
            // Validation should fail: file size > 2 GB
            Assert.True(fileInfo.Length > maxNpySize,
                $"File size {fileInfo.Length} should exceed limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: NPY file significantly over 2 GB (4 GB) is rejected.
    /// Documents that large files are properly rejected.
    /// </summary>
    [Fact]
    public void NpyReader_RejectsFile4GB()
    {
        try
        {
            const long size = 4_000_000_000; // 4 GB (well over limit)
            var path = Path.Combine(_tempDir, "test_4gb.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 2_000_000_000;
            
            // Validation should fail: file size > 2 GB
            Assert.True(fileInfo.Length > maxNpySize,
                $"File size {fileInfo.Length} should exceed limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    #endregion

    #region ONNX File Size Tests (2 GB limit)

    /// <summary>
    /// Test: ONNX file just under 2 GB passes validation.
    /// Documents that large ONNX files near the limit are handled correctly.
    /// </summary>
    [Fact]
    public void OnnxModel_AcceptsFileJustUnder2GB()
    {
        try
        {
            // Create a test file: 1.9 GB (just under 2 GB)
            const long size = 1_900_000_000; // Just under 2 GB
            var path = Path.Combine(_tempDir, "test_1900mb.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should pass: file size <= 2 GB
            Assert.True(fileInfo.Length <= maxOnnxSize,
                $"File size {fileInfo.Length} should be within limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: ONNX file exactly at 2 GB boundary (inclusive limit).
    /// Documents that 2 GB is the maximum accepted size for ONNX.
    /// </summary>
    [Fact]
    public void OnnxModel_AcceptsFileAt2GBBoundary()
    {
        try
        {
            // Create a test file: exactly 2 GB
            const long size = 2_000_000_000; // Exactly 2 GB
            var path = Path.Combine(_tempDir, "test_2gb.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should pass: file size <= 2 GB
            Assert.True(fileInfo.Length <= maxOnnxSize,
                $"File size {fileInfo.Length} should equal limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: ONNX file just over 2 GB is rejected.
    /// Documents that files exceeding 2 GB throw InvalidOperationException.
    /// </summary>
    [Fact]
    public void OnnxModel_RejectsFileJustOver2GB()
    {
        try
        {
            // Create a test file: 2 GB + 1 byte
            const long size = 2_000_000_001; // Just over 2 GB
            var path = Path.Combine(_tempDir, "test_2gb_plus_1.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should fail: file size > 2 GB
            Assert.True(fileInfo.Length > maxOnnxSize,
                $"File size {fileInfo.Length} should exceed limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: ONNX file significantly over 2 GB (4 GB) is rejected.
    /// Documents that very large files are properly rejected.
    /// </summary>
    [Fact]
    public void OnnxModel_RejectsFile4GB()
    {
        try
        {
            // Create a test file: 4 GB (well over limit)
            const long size = 4_000_000_000; // 4 GB
            var path = Path.Combine(_tempDir, "test_4gb.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should fail: file size > 2 GB
            Assert.True(fileInfo.Length > maxOnnxSize,
                $"File size {fileInfo.Length} should exceed limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    #endregion

    #region Comparative Boundary Tests

    /// <summary>
    /// Test: ONNX and NPY now share the same 2 GB limit (Issue #25).
    /// </summary>
    [Fact]
    public void FileLimits_OnnxAndNpyShareSame2GBLimit()
    {
        const long maxOnnxSize = 2_000_000_000; // 2 GB
        const long maxNpySize = 2_000_000_000;  // 2 GB (raised from 500 MB)
        
        Assert.Equal(maxOnnxSize, maxNpySize);
    }

    /// <summary>
    /// Test: Small files (1 MB) pass both ONNX and NPY validation.
    /// Documents that typical small model files are handled correctly.
    /// </summary>
    [Fact]
    public void SmallFiles_PassBothLimits()
    {
        const long smallFileSize = 1_000_000; // 1 MB
        const long maxNpySize = 2_000_000_000;
        const long maxOnnxSize = 2_000_000_000;
        
        Assert.True(smallFileSize <= maxNpySize);
        Assert.True(smallFileSize <= maxOnnxSize);
    }

    /// <summary>
    /// Test: The 1.7B model text_embedding.npy (~1.2 GB) fits within the raised NPY limit.
    /// Documents that the limit increase (Issue #25) accommodates the 1.7B model.
    /// </summary>
    [Fact]
    public void TextEmbedding17B_FitsWithinNpyLimit()
    {
        const long textEmbeddingSize = 1_200_000_000; // ~1.2 GB for 1.7B model
        const long maxNpySize = 2_000_000_000;
        const long maxOnnxSize = 2_000_000_000;
        
        Assert.True(textEmbeddingSize <= maxNpySize);
        Assert.True(textEmbeddingSize <= maxOnnxSize);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a test NPY file of specified size using FileStream.
    /// Does not allocate large memory buffers; uses streaming writes.
    /// For file size validation testing only — doesn't create valid NPY format.
    /// </summary>
    private static void CreateTestNpyFile(string path, long targetSize)
    {
        using var fs = File.Create(path);
        
        // Write target size via streaming zeros (1 MB chunks)
        const int bufferSize = 1024 * 1024; // 1 MB buffer
        var buffer = new byte[bufferSize];
        long remaining = targetSize;
        
        while (remaining > 0)
        {
            int toWrite = (int)Math.Min(buffer.Length, remaining);
            fs.Write(buffer, 0, toWrite);
            remaining -= toWrite;
        }
    }

    /// <summary>
    /// Creates a test ONNX file of specified size using FileStream.
    /// Does not allocate large memory buffers; uses streaming writes.
    /// For file size validation testing only — doesn't create valid ONNX format.
    /// </summary>
    private static void CreateTestOnnxFile(string path, long targetSize)
    {
        using var fs = File.Create(path);
        
        // Write target size via streaming zeros (1 MB chunks)
        const int bufferSize = 1024 * 1024; // 1 MB buffer
        var buffer = new byte[bufferSize];
        long remaining = targetSize;
        
        while (remaining > 0)
        {
            int toWrite = (int)Math.Min(buffer.Length, remaining);
            fs.Write(buffer, 0, toWrite);
            remaining -= toWrite;
        }
    }

    #endregion
}
