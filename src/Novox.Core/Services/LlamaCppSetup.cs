using System.Diagnostics;
using System.Formats.Tar;
using System.IO.Compression;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Novox.Core.Services;

/// <summary>
/// Integración de llama.cpp en Novox, espejo de Clonar-voz (app.py):
/// Clonar-voz NO descarga llama-tts automáticamente; exige instalación
/// manual (winget/brew/releases, build b10500+) y lo AUTODETECTA en PATH
/// + rutas habituales. Este servicio hace lo mismo y además ofrece una
/// descarga opcional desde las releases oficiales de GitHub (tags nocturnas
/// bXXXX, que es donde viven los binarios desde la v0.4.0).
/// </summary>
public sealed class LlamaCppSetup
{
    public sealed record Estado(
        string Binario, bool Existe, string Version, bool SoportaQwen3, string Sugerencia);

    private static readonly HttpClient Api = new() { Timeout = TimeSpan.FromSeconds(60) };
    private static readonly HttpClient Dl = new() { Timeout = Timeout.InfiniteTimeSpan };

    static LlamaCppSetup()
    {
        foreach (var c in new[] { Api, Dl })
            if (!c.DefaultRequestHeaders.Contains("User-Agent"))
                c.DefaultRequestHeaders.Add("User-Agent", "novox/1.0");
    }

    public static string SugerenciaInstalacion()
    {
        if (OperatingSystem.IsWindows())
            return "winget install ggml.llamacpp  (build b10500 o superior)";
        if (OperatingSystem.IsMacOS())
            return "brew install llama.cpp  (build b10500 o superior)";
        return "Descarga el binario de github.com/ggml-org/llama.cpp/releases (build b10500+)";
    }

    public static Estado ObtenerEstado(string? rutaConfigurada = null)
    {
        var bin = string.IsNullOrWhiteSpace(rutaConfigurada) ? BuscarBinario() : rutaConfigurada;
        if (string.IsNullOrEmpty(bin) || !File.Exists(bin))
            return new Estado("", false, "", false, SugerenciaInstalacion());
        var ver = VersionDe(bin);
        var ok = SoportaQwen3(bin);
        return new Estado(bin, true, ver, ok, ok ? "" : "Actualiza: winget upgrade ggml.llamacpp (se necesita --mmproj)");
    }

    public static string BuscarBinario()
    {
        foreach (var n in new[] { "llama-tts", "llama-tts.exe" })
        {
            var p = BuscarEnPath(n);
            if (!string.IsNullOrEmpty(p)) return p;
        }
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        foreach (var c in new[]
        {
            Path.Combine(local, @"Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-tts.exe"),
            "/opt/homebrew/bin/llama-tts", "/usr/local/bin/llama-tts", "/usr/bin/llama-tts",
            @"C:\llama.cpp\llama-tts.exe",
        })
            if (File.Exists(c)) return c;
        try
        {
            var raiz = Path.Combine(local, @"Microsoft\WinGet\Packages");
            if (Directory.Exists(raiz))
            {
                var h = Directory.GetFiles(raiz, "llama-tts.exe", SearchOption.AllDirectories).FirstOrDefault();
                if (!string.IsNullOrEmpty(h)) return h;
            }
        }
        catch { }
        return "";
    }

    public static string VersionDe(string binario)
    {
        var salida = Ejecutar(binario, "--version", 20);
        var m = Regex.Match(salida, @"version:\s*(\S+.*)");
        return m.Success ? m.Groups[1].Value.Trim() : "";
    }

    public static bool SoportaQwen3(string binario) =>
        Ejecutar(binario, "--help", 25).Contains("--mmproj");

    private static string BuscarEnPath(string nombre)
    {
        try
        {
            foreach (var d in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
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

    private static string Ejecutar(string binario, string args, int timeoutSeg)
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo
            {
                FileName = binario,
                Arguments = args,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            })!;
            var so = p.StandardOutput.ReadToEnd();
            var se = p.StandardError.ReadToEnd();
            p.WaitForExit(TimeSpan.FromSeconds(timeoutSeg));
            return so + se;
        }
        catch { return ""; }
    }

    // ---- Auto-descarga opcional desde GitHub releases ----

    private sealed record Activo(
        [property: JsonPropertyName("name")] string name,
        [property: JsonPropertyName("browser_download_url")] string url);
    private sealed record Release(
        [property: JsonPropertyName("tag_name")] string Tag,
        [property: JsonPropertyName("draft")] bool Draft,
        [property: JsonPropertyName("assets")] List<Activo> Assets);

    private static bool EsArchivoBinario(string nombre) =>
        nombre.EndsWith(".zip", StringComparison.OrdinalIgnoreCase)
        || nombre.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Busca la primera release (empezando por latest) que traiga binarios.
    /// Desde la v0.4.0, latest solo trae nightly-tag.txt: los zips viven en
    /// las tags nocturnas bXXXX.
    /// </summary>
    private static async Task<Release> BuscarReleaseConBinariosAsync(CancellationToken ct)
    {
        const string baseUrl = "https://api.github.com/repos/ggml-org/llama.cpp/releases";
        try
        {
            var latest = await Api.GetFromJsonAsync<Release>(baseUrl + "/latest", ct);
            if (latest?.Assets?.Any(a => EsArchivoBinario(a.name)) == true)
                return latest;
        }
        catch { }
        var lista = await Api.GetFromJsonAsync<List<Release>>(baseUrl + "?per_page=20", ct)
            ?? throw new InvalidOperationException("GitHub no devolvió releases.");
        return lista.FirstOrDefault(r => !r.Draft && r.Assets?.Any(a => EsArchivoBinario(a.name)) == true)
            ?? throw new InvalidOperationException(
                "Ninguna release reciente trae binarios. Instálalo manual: " + SugerenciaInstalacion());
    }

    /// <summary>
    /// Descarga el build oficial de llama.cpp para este SO (preferencia GPU:
    /// Vulkan en Windows), extrae llama-tts y lo deja en destinoDir.
    /// Devuelve la ruta del binario.
    /// </summary>
    public static async Task<string> DescargarAsync(
        string destinoDir, IProgress<string>? log = null, CancellationToken ct = default)
    {
        Directory.CreateDirectory(destinoDir);
        log?.Report("Buscando build de llama.cpp con binarios…");
        var rel = await BuscarReleaseConBinariosAsync(ct);
        log?.Report($"Release encontrada: {rel.Tag}");

        var asset = ElegirAsset(rel.Assets.Where(a => EsArchivoBinario(a.name)).Select(a => (a.name, a.url)).ToList());
        if (asset is null)
        {
            var hay = string.Join(", ", rel.Assets.Select(a => a.name).Take(8));
            throw new InvalidOperationException(
                $"La release {rel.Tag} no trae build para este sistema (hay: {hay}). Instálalo manual: " + SugerenciaInstalacion());
        }
        log?.Report($"Descargando {asset.Value.name} ({rel.Tag})…");

        var archivo = Path.Combine(destinoDir, asset.Value.name);
        using (var resp = await Dl.GetAsync(asset.Value.url, HttpCompletionOption.ResponseHeadersRead, ct))
        {
            resp.EnsureSuccessStatusCode();
            var total = resp.Content.Headers.ContentLength ?? 0;
            await using var net = await resp.Content.ReadAsStreamAsync(ct);
            await using var fs = File.Create(archivo);
            var buf = new byte[1 << 20];
            long bajado = 0;
            var t0 = DateTime.UtcNow;
            int n;
            while ((n = await net.ReadAsync(buf, ct)) > 0)
            {
                await fs.WriteAsync(buf.AsMemory(0, n), ct);
                bajado += n;
                if (total > 0 && (DateTime.UtcNow - t0).TotalMilliseconds > 500)
                {
                    t0 = DateTime.UtcNow;
                    log?.Report($"{asset.Value.name}: {100.0 * bajado / total:F1}%");
                }
            }
        }

        log?.Report("Extrayendo llama-tts…");
        var tmp = Path.Combine(destinoDir, "tmp-llama");
        if (Directory.Exists(tmp)) Directory.Delete(tmp, true);
        Directory.CreateDirectory(tmp);
        if (archivo.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
            ZipFile.ExtractToDirectory(archivo, tmp);
        else
        {
            await using var fz = File.OpenRead(archivo);
            using var gz = new GZipStream(fz, CompressionMode.Decompress);
            TarFile.ExtractToDirectory(gz, tmp, true);
        }
        var exe = Directory.GetFiles(tmp, OperatingSystem.IsWindows() ? "llama-tts.exe" : "llama-tts", SearchOption.AllDirectories)
            .FirstOrDefault() ?? throw new InvalidOperationException("El paquete no contiene llama-tts.");
        var final = Path.Combine(destinoDir, Path.GetFileName(exe));
        File.Copy(exe, final, true);
        try { File.Delete(archivo); Directory.Delete(tmp, true); } catch { }
        if (!OperatingSystem.IsWindows())
            try { File.SetUnixFileMode(final, File.GetUnixFileMode(final) | UnixFileMode.UserExecute); } catch { }

        var ver = VersionDe(final);
        log?.Report($"llama-tts instalado: {final} ({ver})");
        if (!SoportaQwen3(final))
            log?.Report("AVISO: este build no anuncia --mmproj; se necesita b10500+.");
        return final;
    }

    private static (string name, string url)? ElegirAsset(List<(string name, string url)> assets)
    {
        string[] prefs;
        if (OperatingSystem.IsWindows())
            prefs = ["vulkan-x64", "cuda-12", "cuda", "win-cpu-x64", "win-x64", "win"];
        else if (OperatingSystem.IsMacOS())
            prefs = [RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "macos-arm64" : "macos-x64", "macos"];
        else
            prefs = ["ubuntu-x64", "linux-x64", "ubuntu", "linux"];
        foreach (var p in prefs)
        {
            var a = assets.FirstOrDefault(x => x.name.Contains(p, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrEmpty(a.name) && !string.IsNullOrEmpty(a.url)) return a;
        }
        var cualquiera = assets.FirstOrDefault(x => !string.IsNullOrEmpty(x.url));
        return string.IsNullOrEmpty(cualquiera.name) ? null : cualquiera;
    }
}
