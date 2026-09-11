namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Configuration options for QwenTTS pipeline.
/// </summary>
public class QwenTtsOptions
{
    /// <summary>
    /// Path to local model directory. When null, uses the variant-specific default shared location.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// HuggingFace repository ID for model download.
    /// When null, automatically determined from <see cref="ModelVariant"/>.
    /// </summary>
    public string? HuggingFaceRepo { get; set; }

    /// <summary>
    /// Model size variant. Defaults to <see cref="QwenModelVariant.Qwen06B"/>.
    /// </summary>
    public QwenModelVariant ModelVariant { get; set; } = QwenModelVariant.Qwen06B;

    /// <summary>
    /// Execution provider for GPU/CPU selection. Default: <see cref="Pipeline.ExecutionProvider.Cpu"/>.
    /// </summary>
    public ExecutionProvider ExecutionProvider { get; set; } = ExecutionProvider.Cpu;

    /// <summary>
    /// GPU device ID when using CUDA or DirectML. Default: 0.
    /// </summary>
    public int GpuDeviceId { get; set; } = 0;

    /// <summary>
    /// Default instruction text for speech style control (e.g., "Read with a calm, warm tone").
    /// Only effective when <see cref="ModelVariant"/> is <see cref="QwenModelVariant.Qwen17B"/> or higher.
    /// Ignored for 0.6B models which do not support instruction control.
    /// </summary>
    public string? InstructText { get; set; }

    /// <summary>
    /// Optional custom session options factory. When set, overrides <see cref="ExecutionProvider"/>.
    /// Use this for advanced scenarios not covered by the enum.
    /// </summary>
    public Func<Microsoft.ML.OnnxRuntime.SessionOptions>? SessionOptionsFactory { get; set; }

    /// <summary>
    /// Maximum number of synthesis requests allowed to run concurrently per pipeline instance.
    /// Default: 1.
    /// </summary>
    public int MaxConcurrency { get; set; } = 1;
}
