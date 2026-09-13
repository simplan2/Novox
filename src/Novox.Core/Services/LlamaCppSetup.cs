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

    /// <summary>
    /// Busca llama-tts: primero en la carpeta que usa Novox
    /// (&lt;modelos&gt;/bin, donde lo deja el botón ⬇), luego PATH y rutas
    /// habituales (winget, brew, /usr, C:\llama.cpp).
    /// </summary>
    public static string BuscarBinario(string? carpetaModelos = null)
    {
        var exe = OperatingSystem.IsWindows() ? "llama-tts.exe" : "llama-tts";
        var dirs = new List<string>();
        try
        {
            var modelos = string.IsNullOrEmpty(carpetaModelos)
                ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "modelos")
                : carpetaModelos;
            dirs.Add(Path.Combine(modelos, "bin"));
            dirs.Add(modelos);
        }
        catch { }
        foreach (var d in dirs)
        {
            try
            {
                var c = Path.Combine(d, exe);
                if (File.Exists(c)) return c;
            }
            catch { }
        }
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

    public sealed record Dispositivo(string Id, string Nombre, long MemoriaMb, long LibreMb);

    /// <summary>Port de listar_dispositivos(): parsea `llama-tts --list-devices`.</summary>
    public static List<Dispositivo> ListarDispositivos(string binario)
    {
        var lista = new List<Dispositivo>();
        if (string.IsNullOrEmpty(binario) || !File.Exists(binario)) return lista;
        var patron = new Regex(@"^\s*([A-Za-z]+\d+):\s+(.+?)\s+\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$");
        foreach (var linea in Ejecutar(binario, "--list-devices", 30).Split('\n'))
        {
            var m = patron.Match(linea.TrimEnd());
            if (m.Success && long.TryParse(m.Groups[3].Value, out var mem) && long.TryParse(m.Groups[4].Value, out var free))
                lista.Add(new Dispositivo(m.Groups[1].Value, m.Groups[2].Value.Trim(), mem, free));
        }
        return lista;
    }

    /// <summary>
    /// Port de dispositivo_preferido(): dedicada (NVIDIA/RTX/Radeon RX/Arc)
    /// sobre integrada; si no, la de más memoria libre. "" = solo CPU.
    /// </summary>
    public static string DispositivoPreferido(string binario)
    {
        var ds = ListarDispositivos(binario);
        if (ds.Count == 0) return "";
        string[] dedicadas = ["nvidia", "geforce", "rtx", "quadro", "tesla", "radeon rx", "arc"];
        foreach (var d in ds)
            if (dedicadas.Any(m => d.Nombre.Contains(m, StringComparison.OrdinalIgnoreCase)))
                return d.Id;
        return ds.MaxBy(d => d.LibreMb)?.Id ?? "";
    }

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
        LimpiarPaquetesViejos(destinoDir, log);
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
        if (Directory.Exists(tmp)) BorrarRecursivo(tmp);
        Directory.CreateDirectory(tmp);
        try
        {
        if (archivo.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
            ExtraerZipSinEnlaces(archivo, tmp, log);
        else
            ExtraerTarGzSinEnlaces(archivo, tmp, log);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new InvalidOperationException(
                $"Windows bloqueó la extracción en '{tmp}' (suele ser por enlaces simbólicos o antivirus). " +
                "Alternativas: 1) winget install ggml.llamacpp  2) activar Modo desarrollador en Windows  3) reintentar. " +
                $"Detalle: {ex.Message}");
        }
        var exe = Directory.GetFiles(tmp, OperatingSystem.IsWindows() ? "llama-tts.exe" : "llama-tts", SearchOption.AllDirectories)
            .FirstOrDefault() ?? throw new InvalidOperationException("El paquete no contiene llama-tts.");
        // llama-tts.exe NO va solo: necesita sus llama.dll / ggml-*.dll hermanas
        // en la misma carpeta (si no, Windows muestra "no se encontró llama.dll").
        var origenDir = Path.GetDirectoryName(exe)!;
        var final = Path.Combine(destinoDir, Path.GetFileName(exe));
        try
        {
            if (File.Exists(final)) File.SetAttributes(final, FileAttributes.Normal);
            File.Copy(exe, final, true);
            log?.Report($"Instalado: {Path.GetFileName(final)}");
            foreach (var dll in Directory.GetFiles(origenDir, "*.dll"))
            {
                var dest = Path.Combine(destinoDir, Path.GetFileName(dll));
                try
                {
                    if (File.Exists(dest)) File.SetAttributes(dest, FileAttributes.Normal);
                    File.Copy(dll, dest, true);
                    log?.Report($"Instalada dependencia: {Path.GetFileName(dll)}");
                }
                catch (Exception ex) { log?.Report($"AVISO: no se pudo copiar {Path.GetFileName(dll)}: {ex.Message}"); }
            }
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new InvalidOperationException(
                $"Sin permiso para escribir en '{destinoDir}'. Cierra programas que lo usen o ejecuta una vez como administrador. " +
                "Alternativa: winget install ggml.llamacpp. " +
                $"Detalle: {ex.Message}");
        }
        try { File.Delete(archivo); BorrarRecursivo(tmp); } catch { }
        if (!OperatingSystem.IsWindows())
            try { File.SetUnixFileMode(final, File.GetUnixFileMode(final) | UnixFileMode.UserExecute); } catch { }

        var ver = VersionDe(final);
        if (string.IsNullOrWhiteSpace(ver))
            throw new InvalidOperationException(
                $"El binario no arranca ({final}). Normalmente faltan DLL hermanas (llama.dll) o el antivirus lo bloqueó. " +
                "Revisa el registro: debe listar las DLL instaladas. Alternativa: winget install ggml.llamacpp.");
        log?.Report($"llama-tts instalado: {final} ({ver})");
        if (!SoportaQwen3(final))
            log?.Report("AVISO: este build no anuncia --mmproj; se necesita b10500+.");
        return final;
    }

    /// <summary>
    /// Extrae un zip omitiendo enlaces simbólicos (crear un symlink en
    /// Windows exige privilegio/modo desarrollador y revienta la
    /// extracción con "el cliente no dispone de un privilegio necesario").
    /// También se protege contra zip-slip.
    /// </summary>
    private static void ExtraerZipSinEnlaces(string zip, string destino, IProgress<string>? log)
    {
        var baseReal = Path.GetFullPath(destino);
        using var arc = ZipFile.OpenRead(zip);
        foreach (var e in arc.Entries)
        {
            if (string.IsNullOrEmpty(e.Name)) continue; // directorio
            if (((e.ExternalAttributes >> 16) & 0xF000) == 0xA000)
            {
                log?.Report($"Omitiendo enlace simbólico: {e.FullName}");
                continue;
            }
            var dest = Path.GetFullPath(Path.Combine(baseReal, e.FullName));
            if (!dest.StartsWith(baseReal, StringComparison.OrdinalIgnoreCase)) continue;
            Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
            e.ExtractToFile(dest, true);
        }
    }

    private static void BorrarRecursivo(string carpeta)
    {
        try
        {
            foreach (var f in Directory.GetFiles(carpeta, "*", SearchOption.AllDirectories))
                try { File.SetAttributes(f, FileAttributes.Normal); } catch { }
            Directory.Delete(carpeta, true);
        }
        catch { }
    }
    private static void ExtraerTarGzSinEnlaces(string archivo, string destino, IProgress<string>? log)
    {
        var baseReal = Path.GetFullPath(destino);
        using var fz = File.OpenRead(archivo);
        using var gz = new GZipStream(fz, CompressionMode.Decompress);
        using var tar = new TarReader(gz);
        TarEntry? e;
        while ((e = tar.GetNextEntry()) is not null)
        {
            if (e.EntryType is not (TarEntryType.RegularFile or TarEntryType.V7RegularFile))
            {
                log?.Report($"Omitiendo: {e.Name} ({e.EntryType})");
                continue;
            }
            var dest = Path.GetFullPath(Path.Combine(baseReal, e.Name));
            if (dest != baseReal && !dest.StartsWith(baseReal + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                continue;
            Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
            e.ExtractToFile(dest, overwrite: true);
        }
    }

    /// <summary>
    /// Borra paquetes de SO equivocado o restos de intentos fallidos
    /// (p. ej. un tarball de Ubuntu en Windows, o tmp-llama a medias).
    /// Solo toca archivos con el patrón oficial llama-*-bin-*.
    /// </summary>
    private static void LimpiarPaquetesViejos(string destinoDir, IProgress<string>? log)
    {
        try
        {
            foreach (var f in Directory.GetFiles(destinoDir, "llama-*-bin-*"))
            {
                try { File.SetAttributes(f, FileAttributes.Normal); File.Delete(f); log?.Report($"Eliminado paquete obsoleto: {Path.GetFileName(f)}"); }
                catch { }
            }
            var tmp = Path.Combine(destinoDir, "tmp-llama");
            if (Directory.Exists(tmp)) BorrarRecursivo(tmp);
        }
        catch { }
    }

    private static (string name, string url)? ElegirAsset(List<(string name, string url)> assets)
    {
        // Filtro ESTRICTO por SO: un Contains("vulkan-x64") suelto llegó a
        // colar un build de Ubuntu en Windows. Primero se exige la marca del
        // SO y solo después se rankea por GPU/formato.
        IEnumerable<(string name, string url)> porSo;
        string[] ranking;
        Func<string, bool> formatoIdeal;
        if (OperatingSystem.IsWindows())
        {
            porSo = assets.Where(x => x.name.Contains("win", StringComparison.OrdinalIgnoreCase));
            ranking = ["vulkan", "cuda", "cpu"];
            formatoIdeal = n => n.EndsWith(".zip", StringComparison.OrdinalIgnoreCase);
        }
        else if (OperatingSystem.IsMacOS())
        {
            porSo = assets.Where(x => x.name.Contains("macos", StringComparison.OrdinalIgnoreCase));
            ranking = [RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "arm64" : "x64"];
            formatoIdeal = n => n.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase);
        }
        else
        {
            porSo = assets.Where(x => x.name.Contains("linux", StringComparison.OrdinalIgnoreCase)
                                   || x.name.Contains("ubuntu", StringComparison.OrdinalIgnoreCase));
            ranking = ["vulkan", "cuda", "cpu"];
            formatoIdeal = n => n.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase);
        }
        var grupo = porSo.Where(x => !string.IsNullOrEmpty(x.url)).ToList();
        if (grupo.Count == 0) return null;
        var ordenados = grupo.Where(x => formatoIdeal(x.name)).Concat(grupo.Where(x => !formatoIdeal(x.name)));
        foreach (var p in ranking)
        {
            var a = ordenados.FirstOrDefault(x => x.name.Contains(p, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrEmpty(a.name)) return a;
        }
        return ordenados.FirstOrDefault();
    }
}
