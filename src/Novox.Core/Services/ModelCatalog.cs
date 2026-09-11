namespace Novox.Core.Services;

/// <summary>
/// Catálogo de modelos GGUF, port de descargar_modelo.py de Clonar-voz.
/// Hacen falta DOS archivos: modelo principal + mmproj (vocoder).
/// </summary>
public static class ModelCatalog
{
    public const string Repo = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF";
    public static string BaseUrl => $"https://huggingface.co/{Repo}/resolve/main";

    public record Entrada(string Clave, string Archivo, long Bytes, string Sha256, string Etiqueta);

    public static readonly IReadOnlyDictionary<string, Entrada> Modelos = new Dictionary<string, Entrada>
    {
        ["Q4_K_M"] = new("Q4_K_M", "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf", 1035965280, "8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129", "Q4_K_M · 1,04 GB · recomendado"),
        ["Q8_0"] = new("Q8_0", "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", 1847874400, "ac7931aeb2e7aad1a6ed6602d353a5679c9d096b18ce8204ac730a8408d572e1", "Q8_0 · 1,85 GB · más calidad"),
        ["bf16"] = new("bf16", "Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf", 3472593760, "0322c634ad5d3282524bc45bff030ab3f6f2a32ba14d5e7dace7eed75ecede46", "BF16 · 3,47 GB · sin cuantizar"),
    };

    public static readonly IReadOnlyDictionary<string, Entrada> Mmproj = new Dictionary<string, Entrada>
    {
        ["Q8_0"] = new("Q8_0", "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", 446422912, "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2", "Q8_0 · 446 MB · recomendado"),
        ["bf16"] = new("bf16", "mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf", 669081472, "b9503e95e44705739cf82c15ce909b040b96a31ad79d02cf4dc5e1484399265d", "BF16 · 669 MB · sin cuantizar"),
    };

    public const string PorDefectoModelo = "Q4_K_M";
    public const string PorDefectoMmproj = "Q8_0";

    /// <summary>True si falta el par modelo+mmproj utilizable (port de falta_algo).</summary>
    public static bool FaltaAlgo(string carpeta)
    {
        if (!Directory.Exists(carpeta)) return true;
        var ggufs = Directory.GetFiles(carpeta, "*.gguf");
        return !(ggufs.Any(g => Path.GetFileName(g).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase))
              && ggufs.Any(g => !Path.GetFileName(g).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase)));
    }
}
