using Novox.Core.Services.Inference;

namespace Novox.Core.Services;

/// <summary>
/// Backend ONNX (ElBruno.QwenTTS.VoiceCloning): inferencia 100% .NET con
/// ONNX Runtime, sin llama.cpp ni ffmpeg ni Python.
/// Se resuelve por reflexión para no exigir los paquetes NuGet
/// (Microsoft.ML.OnnxRuntime, ~100 MB, + ~5.5 GB de modelos) al compilar:
/// si no están instalados, lanza un error explicativo en vez de romper.
/// </summary>
public sealed class OnnxBlockSynthesizer : IBlockSynthesizer
{
    public string Nombre => "onnx";

    public static bool EnsambladoDisponible()
    {
        try
        {
            return AppDomain.CurrentDomain.GetAssemblies()
                .Any(a => a.GetName().Name?.StartsWith("ElBruno.QwenTTS") == true)
                || Type.GetType("ElBruno.QwenTTS.VoiceCloning.Pipeline.VoiceClonePipeline, ElBruno.QwenTTS.VoiceCloning") is not null;
        }
        catch { return false; }
    }

    public async Task SynthesizeBlockAsync(string bloque, string salidaWav, string? vozRefPath,
        BlockOptions opts, IProgress<string>? log = null, CancellationToken ct = default)
    {
        if (string.IsNullOrEmpty(vozRefPath) || !File.Exists(vozRefPath))
            throw new InvalidOperationException("[onnx] Este backend exige voz de referencia: elige una voz de la biblioteca.");
        if (!EnsambladoDisponible())
            throw new InvalidOperationException(
                "[onnx] Paquete ElBruno.QwenTTS.VoiceCloning no instalado. " +
                "Ejecuta: dotnet add package ElBruno.QwenTTS.VoiceCloning " +
                "(descarga ~5.5 GB del modelo Base la primera vez).");

        log?.Report("[onnx] sintetizando bloque con VoiceClonePipeline…");
        // Resolución por reflexión: CreateAsync() → SynthesizeAsync(text, ref, out, lang)
        var tipo = Type.GetType("ElBruno.QwenTTS.VoiceCloning.Pipeline.VoiceClonePipeline, ElBruno.QwenTTS.VoiceCloning")
            ?? throw new InvalidOperationException("[onnx] No se pudo resolver VoiceClonePipeline.");
        var crear = tipo.GetMethod("CreateAsync", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)
            ?? throw new InvalidOperationException("[onnx] API incompatible: falta CreateAsync.");
        using var _ = ct.Register(() => { });
        dynamic cloner = await (dynamic)crear.Invoke(null, null)!;
        await (Task)tipo.GetMethod("SynthesizeAsync")!.Invoke(cloner,
            new object[] { bloque, vozRefPath, salidaWav, MapLanguage(opts.Language) })!;
        if (!File.Exists(salidaWav)) throw new InvalidOperationException("[onnx] El pipeline no generó audio.");
    }

    private static string MapLanguage(string iso) => iso switch
    {
        "es" => "spanish", "en" => "english", "zh" => "chinese",
        "ja" => "japanese", "ko" => "korean", "ru" => "russian",
        _ => "english",
    };
}
