using Novox.Core.Models;
using Novox.Core.Services.Inference;

namespace Novox.Core.Services;

public interface ITtsService
{
    Task InitializeAsync(IProgress<string>? progress = null, CancellationToken ct = default);
    Task<GeneracionResultado> SynthesizeAsync(string text, string speaker, string language, CancellationToken ct = default);
    Task<GeneracionResultado> SynthesizeAsync(string text, string? vozRefPath, string speaker, string language, IProgress<ProgresoInfo>? progress = null, CancellationToken ct = default);
    Task<ReadOnlyMemory<byte>> SynthesizeToWavAsync(string text, string speaker, string language, CancellationToken ct = default);
    IReadOnlyList<string> GetAvailableSpeakers();
    IReadOnlyList<string> ListDevices();
    string PreferredDevice();
    bool IsInitialized { get; }
    bool SupportsGpu { get; }
    void Dispose();
}

/// <summary>
/// Motor TTS de Novox, con el mismo flujo que Clonar-voz (app.py):
/// trocear por bloques → sintetizar bloque a bloque con progreso/log →
/// unir con pausa intermedia. Solo una síntesis a la vez (cerrojo).
/// El backend de inferencia es intercambiable: hoy genera audio local
/// (silencio proporcional, para validar el pipeline de punta a punta);
/// el punto de inserción del backend real (llama.cpp o ElBruno.QwenTTS
/// ONNX) es <see cref="SintetizarBloqueAsync"/>.
/// </summary>
public class TtsService : ITtsService
{
    private bool _initialized;
    private readonly AppConfig _config;
    private readonly string _modelDir;
    private readonly string _outDir;
    private static readonly SemaphoreSlim Cerrojo = new(1, 1);

    public bool IsInitialized => _initialized;
    public bool SupportsGpu => true;
    public string BackendActual { get; private set; } = "local";

    public TtsService(AppConfig config)
    {
        _config = config;
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        _modelDir = string.IsNullOrEmpty(config.ModelDirectory) ? Path.Combine(baseDir, "modelos") : config.ModelDirectory;
        _outDir = string.IsNullOrEmpty(config.OutputsDirectory) ? Path.Combine(baseDir, "salidas") : config.OutputsDirectory;
    }

    public async Task InitializeAsync(IProgress<string>? progress = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        progress?.Report("Verificando modelo Qwen3-TTS...");
        await Task.Delay(50, ct);
        Directory.CreateDirectory(_modelDir);
        Directory.CreateDirectory(_outDir);
        _initialized = true;
        BackendActual = ResolverBackend(_config.Motor);
        progress?.Report(ModelCatalog.FaltaAlgo(_modelDir)
            ? $"Motor listo (backend: {BackendActual}; sin modelo GGUF: usa la pestaña de descarga)."
            : $"Motor Qwen3-TTS listo (backend: {BackendActual}).");
    }

    public Task<GeneracionResultado> SynthesizeAsync(string text, string speaker, string language, CancellationToken ct = default)
        => SynthesizeAsync(text, null, speaker, language, null, ct);

    public async Task<GeneracionResultado> SynthesizeAsync(
        string text, string? vozRefPath, string speaker, string language,
        IProgress<ProgresoInfo>? progress = null, CancellationToken ct = default)
    {
        if (!_initialized) throw new InvalidOperationException("El motor no está inicializado. Llama InitializeAsync primero.");
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Texto vacío.", nameof(text));

        var bloques = TextChunker.Trocear(text, _config.CaracteresPorBloque > 0 ? _config.CaracteresPorBloque : 280);
        var id = Guid.NewGuid().ToString("N")[..12];
        var fecha = DateTime.Now;
        var nombreArchivo = $"{fecha:yyyyMMdd-HHmmss}-{id}.wav";
        var destino = Path.Combine(_outDir, nombreArchivo);
        var dispositivo = ResolverDispositivo(_config.Dispositivo);
        var tmp = Path.Combine(Path.GetTempPath(), "novox", id);
        Directory.CreateDirectory(tmp);

        var inicio = DateTime.UtcNow;
        var partes = new List<string>();
        var adquirido = false;
        try
        {
            if (!await Cerrojo.WaitAsync(0, ct))
            {
                progress?.Report(new ProgresoInfo { Estado = "log", Mensaje = "Hay otra síntesis en curso; esperando turno…" });
                await Cerrojo.WaitAsync(ct);
            }
            adquirido = true;
            BloqueLog = m => progress?.Report(new ProgresoInfo { Estado = "log", Mensaje = m });

            for (var i = 0; i < bloques.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                var parcial = Path.Combine(tmp, $"{id}_{i + 1:000}.wav");
                progress?.Report(new ProgresoInfo
                {
                    Estado = "bloque",
                    BloqueActual = i + 1,
                    TotalBloques = bloques.Count,
                    Porcentaje = bloques.Count == 0 ? 100 : 100.0 * i / bloques.Count,
                    TextoActual = bloques[i],
                    Mensaje = $"Bloque {i + 1}/{bloques.Count}"
                });
                await SintetizarBloqueAsync(bloques[i], parcial, vozRefPath, language, ct);
                progress?.Report(new ProgresoInfo { Estado = "log", Mensaje = $"bloque {i + 1}/{bloques.Count} OK ({Path.GetFileName(parcial)})" });
                partes.Add(parcial);
            }

            ct.ThrowIfCancellationRequested();
            WavUtils.Unir(partes.ToArray(), destino, _config.PausaMs);
            var dur = WavUtils.Duracion(destino);
            var seg = (DateTime.UtcNow - inicio).TotalSeconds;
            progress?.Report(new ProgresoInfo { Estado = "fin", BloqueActual = bloques.Count, TotalBloques = bloques.Count, Porcentaje = 100 });

            return new GeneracionResultado
            {
                Id = id,
                RutaArchivo = destino,
                NombreArchivo = nombreArchivo,
                Duracion = Math.Round(dur, 2),
                SegundosGeneracion = Math.Round(seg, 1),
                Dispositivo = string.IsNullOrEmpty(dispositivo) ? "CPU" : dispositivo,
                Idioma = language,
                Texto = text,
                Fecha = fecha,
                VozId = speaker
            };
        }
        finally
        {
            if (adquirido) Cerrojo.Release();
            try { foreach (var p in partes) File.Delete(p); Directory.Delete(tmp, true); } catch { }
        }
    }

    /// <summary>
    /// Punto de inserción del backend real, seleccionable:
    /// "llama" (GGUF 1.7B, como Clonar-voz), "onnx" (ElBruno.QwenTTS
    /// VoiceCloning, 100% .NET), "local" (validación del pipeline),
    /// "auto" (llama si hay binario+modelos, si no onnx si está instalado,
    /// si no local).
    /// </summary>
    protected virtual async Task SintetizarBloqueAsync(
        string bloque, string salidaWav, string? vozRefPath, string language, CancellationToken ct)
    {
        var (binario, modelo, mmproj) = RutasEfectivas();
        var backend = CrearBackend(ResolverBackend(_config.Motor, binario, modelo, mmproj));
        BackendActual = backend.Nombre;
        var log = new Progress<string>(m => BloqueLog?.Invoke(m));
        await backend.SynthesizeBlockAsync(bloque, salidaWav, vozRefPath, new BlockOptions(
            _config.Temp, _config.TopP, _config.TopK, _config.Semilla, _config.MaxFrames,
            _config.Hilos, _config.CapasGpu, ResolverDispositivo(_config.Dispositivo),
            language, binario, modelo, mmproj), log, ct);
    }

    /// <summary>Log de bajo nivel del backend (el servicio lo reemite como ProgresoInfo).</summary>
    public Action<string>? BloqueLog { get; set; }

    private static IBlockSynthesizer CrearBackend(string nombre) => nombre switch
    {
        "llama" => new LlamaCppBlockSynthesizer(),
        "onnx" => new OnnxBlockSynthesizer(),
        _ => new SilentBlockSynthesizer(),
    };

    private (string binario, string modelo, string mmproj) RutasEfectivas()
    {
        var binario = _config.Binario;
        if (string.IsNullOrEmpty(binario)) binario = BuscarBinario();
        var modelo = _config.Modelo;
        var mmproj = _config.Mmproj;
        if (string.IsNullOrEmpty(modelo) || string.IsNullOrEmpty(mmproj))
        {
            var (autoModelo, autoMmproj) = BuscarModelos();
            modelo = string.IsNullOrEmpty(modelo) ? autoModelo : modelo;
            mmproj = string.IsNullOrEmpty(mmproj) ? autoMmproj : mmproj;
        }
        return (binario, modelo, mmproj);
    }

    private static string ResolverBackend(string motor, string binario = "", string modelo = "", string mmproj = "")
    {
        motor = (motor ?? "auto").Trim().ToLowerInvariant();
        if (motor is "llama" or "onnx" or "local") return motor;
        if (!string.IsNullOrEmpty(binario) && LlamaCppBlockSynthesizer.Disponible(binario, modelo, mmproj)) return "llama";
        if (OnnxBlockSynthesizer.EnsambladoDisponible()) return "onnx";
        if (LlamaCppBlockSynthesizer.Disponible(binario, modelo, mmproj)) return "llama";
        return "local";
    }

    public async Task<ReadOnlyMemory<byte>> SynthesizeToWavAsync(string text, string speaker, string language, CancellationToken ct = default)
    {
        var r = await SynthesizeAsync(text, speaker, language, ct);
        return File.Exists(r.RutaArchivo) ? await File.ReadAllBytesAsync(r.RutaArchivo, ct) : Array.Empty<byte>();
    }

    public IReadOnlyList<string> GetAvailableSpeakers() =>
        new[] { "ryan", "serena", "vivian", "aiden", "eric", "dylan", "uncle_fu", "ono_anna", "sohee" };

    public IReadOnlyList<string> ListDevices() => new[] { "CPU", "auto", "Vulkan0", "Vulkan1" };

    public string PreferredDevice() => "auto";

    private static string ResolverDispositivo(string modo)
    {
        if (string.IsNullOrWhiteSpace(modo)) return "";
        modo = modo.Trim().ToLowerInvariant();
        if (modo is "cpu" or "none" or "solo cpu") return "";
        if (modo is "auto" or "automático" or "automatico") return "auto";
        return modo;
    }

    /// <summary>Port de buscar_binario() de Clonar-voz.</summary>
    private static string BuscarBinario()
    {
        foreach (var n in new[] { "llama-tts", "llama-tts.exe" })
        {
            var enPath = BuscarEnPath(n);
            if (!string.IsNullOrEmpty(enPath)) return enPath;
        }
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var candidatos = new[]
        {
            Path.Combine(local, @"Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-tts.exe"),
            "/opt/homebrew/bin/llama-tts", "/usr/local/bin/llama-tts", "/usr/bin/llama-tts",
            "C:\\llama.cpp\\llama-tts.exe",
        };
        foreach (var c in candidatos) if (File.Exists(c)) return c;
        try
        {
            var raiz = Path.Combine(local, @"Microsoft\WinGet\Packages");
            if (Directory.Exists(raiz))
            {
                var hallado = Directory.GetFiles(raiz, "llama-tts.exe", SearchOption.AllDirectories).FirstOrDefault();
                if (!string.IsNullOrEmpty(hallado)) return hallado;
            }
        }
        catch { }
        return "";
    }

    private static string BuscarEnPath(string nombre)
    {
        try
        {
            var paths = (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator);
            foreach (var d in paths)
            {
                try
                {
                    var c = Path.Combine(d.Trim(), nombre);
                    if (File.Exists(c)) return Path.GetFullPath(c);
                }
                catch { }
            }
        }
        catch { }
        return "";
    }

    /// <summary>Port de buscar_modelos() de Clonar-voz.</summary>
    private (string modelo, string mmproj) BuscarModelos()
    {
        try
        {
            if (!Directory.Exists(_modelDir)) return ("", "");
            var ggufs = Directory.GetFiles(_modelDir, "*.gguf");
            var mm = ggufs.FirstOrDefault(g => Path.GetFileName(g).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase)) ?? "";
            var mo = ggufs.FirstOrDefault(g => !Path.GetFileName(g).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase)) ?? "";
            return (mo, mm);
        }
        catch { return ("", ""); }
    }

    public void Dispose() { }
}
