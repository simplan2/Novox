using System.Security.Cryptography;
using Novox.Core.Models;

namespace Novox.Core.Services;

public interface IModelDownloadService
{
    event Action<string>? OnProgress;
    event Action<double>? OnProgressPercent;
    event Action? OnComplete;
    event Action<string>? OnError;
    bool IsDownloading { get; }
    double Progress { get; set; }
    string? CurrentFile { get; set; }
    Task DownloadModelsAsync(string variant = "Qwen17B", CancellationToken ct = default);
    Task DownloadAsync(string claveModelo, string claveMmproj, CancellationToken ct = default);
    void Cancel();
    bool IsModelAvailable();
}

/// <summary>
/// Descarga reanudable (HTTP Range) + verificación SHA-256, port de
/// descargar_modelo.py de Clonar-voz. Descarga el par modelo+mmproj.
/// </summary>
public class ModelDownloadService : IModelDownloadService
{
    private CancellationTokenSource? _cts;
    private bool _isDownloading;
    private readonly string _modelDir;
    private static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    public event Action<string>? OnProgress;
    public event Action<double>? OnProgressPercent;
    public event Action? OnComplete;
    public event Action<string>? OnError;
    public bool IsDownloading => _isDownloading;
    public double Progress { get; set; }
    public string? CurrentFile { get; set; }

    public ModelDownloadService(AppConfig config)
    {
        _modelDir = string.IsNullOrEmpty(config.ModelDirectory)
            ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "modelos")
            : config.ModelDirectory;
        if (!Http.DefaultRequestHeaders.Contains("User-Agent"))
            Http.DefaultRequestHeaders.Add("User-Agent", "novox/1.0");
    }

    public Task DownloadModelsAsync(string variant = "Qwen17B", CancellationToken ct = default)
    {
        var (m, p) = variant switch
        {
            "Q8_0" => ("Q8_0", "Q8_0"),
            "bf16" => ("bf16", "bf16"),
            _ => (ModelCatalog.PorDefectoModelo, ModelCatalog.PorDefectoMmproj),
        };
        return DownloadAsync(m, p, ct);
    }

    public async Task DownloadAsync(string claveModelo, string claveMmproj, CancellationToken ct = default)
    {
        _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var tok = _cts.Token;
        _isDownloading = true;
        try
        {
            Directory.CreateDirectory(_modelDir);
            if (!ModelCatalog.Modelos.TryGetValue(claveModelo, out var m)) throw new ArgumentException($"Modelo desconocido: {claveModelo}");
            if (!ModelCatalog.Mmproj.TryGetValue(claveMmproj, out var p)) throw new ArgumentException($"mmproj desconocido: {claveMmproj}");
            await DescargarEntrada(m, tok);
            await DescargarEntrada(p, tok);
            OnComplete?.Invoke();
        }
        catch (OperationCanceledException) { OnError?.Invoke("Descarga cancelada."); }
        catch (Exception ex) { OnError?.Invoke(ex.Message); }
        finally { _isDownloading = false; }
    }

    private async Task DescargarEntrada(ModelCatalog.Entrada e, CancellationToken ct)
    {
        CurrentFile = e.Archivo;
        var destino = Path.Combine(_modelDir, e.Archivo);
        var parcial = destino + ".parte";
        if (File.Exists(destino) && new FileInfo(destino).Length == e.Bytes)
        {
            OnProgress?.Invoke($"{e.Archivo}: ya descargado.");
            return;
        }

        // El parcial puede estar COMPLETO de un intento anterior (tamaño exacto
        // pero sin verificar): en ese caso se verifica y se da por bueno en vez
        // de pedir Range más allá del EOF (el servidor responde 416).
        if (File.Exists(parcial) && new FileInfo(parcial).Length == e.Bytes)
        {
            OnProgress?.Invoke($"{e.Archivo}: comprobando descarga previa…");
            if (await ShaOkAsync(parcial, e.Sha256, e.Archivo))
            {
                File.Move(parcial, destino, true);
                OnProgress?.Invoke($"{e.Archivo}: OK (verificado).");
                return;
            }
            File.Delete(parcial);
        }
        // Bucle de 2 intentos: si el servidor rechaza el Range con 416
        // (p. ej. parcial corrupto por arriba del tamaño), se reintenta
        // desde cero en vez de morir y borrar lo descargado sin más.
        for (var intento = 0; intento < 2; intento++)
        {
            long desde = File.Exists(parcial) ? new FileInfo(parcial).Length : 0;
            if (desde > e.Bytes) { File.Delete(parcial); desde = 0; }

            using var req = new HttpRequestMessage(HttpMethod.Get, $"{ModelCatalog.BaseUrl}/{e.Archivo}?download=true");
            if (desde > 0) req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(desde, null);
            using var resp = await Http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            if (resp.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
            {
                try { if (File.Exists(parcial)) File.Delete(parcial); } catch { }
                continue; // reintenta sin Range
            }
            if (desde > 0 && resp.StatusCode != System.Net.HttpStatusCode.PartialContent)
            {
                // El servidor ignoró el rango: empezar de cero.
                try { if (File.Exists(parcial)) File.Delete(parcial); } catch { }
                desde = 0;
            }
            resp.EnsureSuccessStatusCode();

            await using var net = await resp.Content.ReadAsStreamAsync(ct);
            await using var fs = new FileStream(parcial, desde > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None);
            var buf = new byte[1 << 20];
            var descargado = desde;
            int n;
            var t0 = DateTime.UtcNow;
            while ((n = await net.ReadAsync(buf, ct)) > 0)
            {
                await fs.WriteAsync(buf.AsMemory(0, n), ct);
                descargado += n;
                var pct = 100.0 * descargado / e.Bytes;
                Progress = pct;
                OnProgressPercent?.Invoke(pct);
                if ((DateTime.UtcNow - t0).TotalMilliseconds > 300)
                {
                    t0 = DateTime.UtcNow;
                    OnProgress?.Invoke($"{e.Archivo}: {pct:F1}%");
                }
            }
            break;
        }
        if (!File.Exists(parcial) || new FileInfo(parcial).Length != e.Bytes)
            throw new InvalidOperationException($"{e.Archivo}: descarga incompleta. Vuelve a pulsar Descargar para reanudarla.");

        if (!await ShaOkAsync(parcial, e.Sha256, e.Archivo))
        {
            try { File.Delete(parcial); } catch { }
            throw new InvalidOperationException($"{e.Archivo}: el SHA-256 no coincide (archivo corrupto).");
        }
        File.Move(parcial, destino, true);
        OnProgress?.Invoke($"{e.Archivo}: OK.");
    }

    private async Task<bool> ShaOkAsync(string ruta, string esperado, string nombre)
    {
        OnProgress?.Invoke($"Verificando SHA-256 de {nombre}…");
        using var sha = SHA256.Create();
        await using var fr = File.OpenRead(ruta);
        var hash = Convert.ToHexString(await sha.ComputeHashAsync(fr)).ToLowerInvariant();
        return hash == esperado.ToLowerInvariant();
    }

    public void Cancel() => _cts?.Cancel();
    public bool IsModelAvailable() => !ModelCatalog.FaltaAlgo(_modelDir);
}
