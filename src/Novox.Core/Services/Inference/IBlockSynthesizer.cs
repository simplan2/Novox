namespace Novox.Core.Services.Inference;

/// <summary>Opciones por bloque, espejo de construir_comando() de Clonar-voz.</summary>
public sealed record BlockOptions(
    double Temp,
    double TopP,
    int TopK,
    int Semilla,
    int MaxFrames,
    int Hilos,
    int CapasGpu,
    string Dispositivo,   // "" = CPU, "auto" = preferida, o id (Vulkan1…)
    string Language,
    string BinarioPath,
    string ModeloPath,
    string MmprojPath);

/// <summary>Backend de inferencia intercambiable: sintetiza UN bloque a WAV.</summary>
public interface IBlockSynthesizer
{
    string Nombre { get; }
    Task SynthesizeBlockAsync(
        string bloque, string salidaWav, string? vozRefPath,
        BlockOptions opts, IProgress<string>? log = null, CancellationToken ct = default);
}
