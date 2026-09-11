using Novox.Core.Services.Inference;

namespace Novox.Core.Services;

/// <summary>
/// Backend "local": genera audio proporcional al texto para validar el
/// pipeline (troceado → bloques → unión → historial) sin modelo.
/// </summary>
public sealed class SilentBlockSynthesizer : IBlockSynthesizer
{
    public string Nombre => "local";
    public async Task SynthesizeBlockAsync(string bloque, string salidaWav, string? vozRefPath,
        BlockOptions opts, IProgress<string>? log = null, CancellationToken ct = default)
    {
        var segundos = Math.Clamp(bloque.Length / 14.0, 0.4, 90.0);
        log?.Report($"[local] bloque de {bloque.Length} car → {segundos:F1}s (sin modelo)");
        await Task.Delay(TimeSpan.FromMilliseconds(Math.Min(400, segundos * 60)), ct);
        WavUtils.EscribirSilencio(salidaWav, segundos);
    }
}
