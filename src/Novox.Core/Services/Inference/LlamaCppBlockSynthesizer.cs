using System.Diagnostics;
using Novox.Core.Services.Inference;

namespace Novox.Core.Services;

/// <summary>
/// Backend llama.cpp: réplica de construir_comando() de Clonar-voz.
/// Ejecuta llama-tts por bloque con -mmdev (decisivo en GPU: sin él el
/// vocoder corre en CPU y tarda ~15x más).
/// </summary>
public sealed class LlamaCppBlockSynthesizer : IBlockSynthesizer
{
    public string Nombre => "llama";

    public static bool Disponible(string binario, string modelo, string mmproj) =>
        File.Exists(binario) && File.Exists(modelo) && File.Exists(mmproj);

    public async Task SynthesizeBlockAsync(string bloque, string salidaWav, string? vozRefPath,
        BlockOptions opts, IProgress<string>? log = null, CancellationToken ct = default)
    {
        if (!File.Exists(opts.BinarioPath)) throw new FileNotFoundException("No se encuentra llama-tts. Instálalo (winget install ggml.llamacpp) o pon la ruta en config.", opts.BinarioPath);
        if (!File.Exists(opts.ModeloPath)) throw new FileNotFoundException("Falta el modelo .gguf principal.", opts.ModeloPath);
        if (!File.Exists(opts.MmprojPath)) throw new FileNotFoundException("Falta el mmproj (vocoder).", opts.MmprojPath);

        var cmd = new List<string>
        {
            opts.BinarioPath, "-m", opts.ModeloPath, "-mm", opts.MmprojPath,
            "-p", bloque, "-o", salidaWav,
            "--tts-lang", opts.Language, "-n", opts.MaxFrames.ToString(),
            "--top-k", opts.TopK.ToString(), "--top-p", opts.TopP.ToString(),
            "--temp", opts.Temp.ToString(System.Globalization.CultureInfo.InvariantCulture),
        };
        if (string.IsNullOrEmpty(opts.Dispositivo))
            cmd.AddRange(["-ngl", "0", "--device", "none", "--no-mmproj-offload"]);
        else
            cmd.AddRange(["-ngl", opts.CapasGpu.ToString(), "--device", opts.Dispositivo, "-mmdev", opts.Dispositivo]);
        if (opts.Hilos > 0) cmd.AddRange(["-t", opts.Hilos.ToString()]);
        if (opts.Semilla >= 0) cmd.AddRange(["-s", opts.Semilla.ToString()]);
        if (!string.IsNullOrEmpty(vozRefPath)) cmd.AddRange(["--tts-speaker-file", vozRefPath]);

        log?.Report(string.IsNullOrEmpty(vozRefPath)
            ? "[llama] SIN referencia → voz por defecto del modelo (no es clonación)"
            : "[llama] clonando con referencia: " + vozRefPath);
        using var proc = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = cmd[0],
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            }
        };
        foreach (var a in cmd.Skip(1)) proc.StartInfo.ArgumentList.Add(a);
        proc.Start();
        using (ct.Register(() => { try { if (!proc.HasExited) proc.Kill(true); } catch { } }))
        {
            proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) log?.Report("[llama] " + e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) log?.Report("[llama] " + e.Data); };
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            await proc.WaitForExitAsync(ct);
        }
        if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
        if (proc.ExitCode != 0 || !File.Exists(salidaWav))
            throw new InvalidOperationException($"llama-tts terminó con código {proc.ExitCode}. Revisa el registro.");
    }
}
