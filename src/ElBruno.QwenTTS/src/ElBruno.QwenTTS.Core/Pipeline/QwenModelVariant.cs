namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Identifies a Qwen3-TTS model size variant.
/// </summary>
public enum QwenModelVariant
{
    /// <summary>0.6B parameter model (default). Talker hidden_size=1024.</summary>
    Qwen06B = 0,

    /// <summary>1.7B parameter model. Talker hidden_size=2048. Supports instruction control.</summary>
    Qwen17B = 1,
}

/// <summary>
/// Provides configuration values that differ between Qwen model variants.
/// At download time these values select the correct HuggingFace repo and
/// local storage path. At runtime the actual dimensions come from <c>config.json</c>.
/// </summary>
public static class QwenModelVariantConfig
{
    /// <summary>Default variant when none is specified.</summary>
    public const QwenModelVariant Default = QwenModelVariant.Qwen06B;

    /// <summary>Returns the expected Talker LM hidden size for a variant.</summary>
    public static int GetHiddenSize(QwenModelVariant variant) => variant switch
    {
        QwenModelVariant.Qwen06B => 1024,
        QwenModelVariant.Qwen17B => 2048,
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown model variant")
    };

    /// <summary>Returns the expected Talker LM intermediate (MLP) size for a variant.</summary>
    public static int GetIntermediateSize(QwenModelVariant variant) => variant switch
    {
        QwenModelVariant.Qwen06B => 3072,
        QwenModelVariant.Qwen17B => 6144,
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown model variant")
    };

    /// <summary>Returns the HuggingFace repository ID for a variant.</summary>
    public static string GetRepoId(QwenModelVariant variant) => variant switch
    {
        QwenModelVariant.Qwen06B => "elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX",
        QwenModelVariant.Qwen17B => "elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX",
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown model variant")
    };

    /// <summary>
    /// Returns the model subdirectory name for a variant.
    /// Used to keep different model files separate under the shared cache root.
    /// </summary>
    public static string GetModelSubDir(QwenModelVariant variant) => variant switch
    {
        QwenModelVariant.Qwen06B => "0.6B",
        QwenModelVariant.Qwen17B => "1.7B",
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown model variant")
    };

    /// <summary>
    /// Returns the full default model directory for a variant.
    /// The 0.6B variant uses the legacy root path (backward compatible).
    /// Other variants use a size-specific subdirectory.
    /// </summary>
    public static string GetDefaultModelDir(QwenModelVariant variant) => variant switch
    {
        QwenModelVariant.Qwen06B => ModelDownloader.DefaultModelDir,
        QwenModelVariant.Qwen17B => Path.Combine(ModelDownloader.DefaultModelDir, "1.7B"),
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown model variant")
    };

    /// <summary>Returns whether a variant supports instruction control (emotion, rate, timbre).</summary>
    public static bool SupportsInstruct(QwenModelVariant variant) => variant switch
    {
        QwenModelVariant.Qwen06B => false,
        QwenModelVariant.Qwen17B => true,
        _ => false
    };

    /// <summary>Returns all defined model variants.</summary>
    public static QwenModelVariant[] GetAllVariants() =>
        [QwenModelVariant.Qwen06B, QwenModelVariant.Qwen17B];
}
