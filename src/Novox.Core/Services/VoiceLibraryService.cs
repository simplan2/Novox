using System.Text.Json;
using Novox.Core.Models;

namespace Novox.Core.Services;

public interface IVoiceLibraryService
{
    Task<List<VozInfo>> GetVoicesAsync();
    Task<VozInfo> AddVoiceAsync(string nombre, string transcripcion, MemoryStream audioStream);
    Task DeleteVoiceAsync(string id);
    Task<byte[]> GetVoiceAudioAsync(string id);
}

public class VoiceLibraryService : IVoiceLibraryService
{
    private readonly AppConfig _config;
    private readonly string _vocesDir;

    public VoiceLibraryService(AppConfig config)
    {
        _config = config;
        _vocesDir = string.IsNullOrEmpty(config.VoicesDirectory)
            ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "voces")
            : config.VoicesDirectory;
        Directory.CreateDirectory(_vocesDir);
    }

    public async Task<List<VozInfo>> GetVoicesAsync()
    {
        var voces = new List<VozInfo>();
        if (!Directory.Exists(_vocesDir)) return voces;

        foreach (var carpeta in Directory.GetDirectories(_vocesDir))
        {
            try
            {
                var metaPath = Path.Combine(carpeta, "voz.json");
                var audioPath = Path.Combine(carpeta, "referencia.wav");
                if (!File.Exists(metaPath) || !File.Exists(audioPath)) continue;

                var json = await File.ReadAllTextAsync(metaPath);
                var meta = JsonSerializer.Deserialize<VozInfo>(json)!;
                meta.Id = Path.GetFileName(carpeta);
                meta.RutaAudio = audioPath;
                meta.Creada = meta.Creada == default ? File.GetCreationTimeUtc(carpeta) : meta.Creada;

                try { meta.Duracion = GetDuration(audioPath); } catch { meta.Duracion = 0; }

                voces.Add(meta);
            }
            catch { }
        }

        return voces.OrderByDescending(v => v.Creada).ToList();
    }

    public async Task<VozInfo> AddVoiceAsync(string nombre, string transcripcion, MemoryStream audioStream)
    {
        var id = Guid.NewGuid().ToString("N")[..12];
        var carpeta = Path.Combine(_vocesDir, id);
        Directory.CreateDirectory(carpeta);

        var audioPath = Path.Combine(carpeta, "referencia.wav");
        audioStream.Position = 0;
        await using (var file = File.Create(audioPath))
        {
            audioStream.CopyTo(file);
        }

        var meta = new VozInfo
        {
            Id = id,
            Nombre = nombre,
            RutaAudio = audioPath,
            Transcripcion = transcripcion,
            Duracion = GetDuration(audioPath),
            Creada = DateTime.UtcNow
        };

        await File.WriteAllTextAsync(Path.Combine(carpeta, "voz.json"),
            JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));

        return meta;
    }

    public Task DeleteVoiceAsync(string id)
    {
        var carpeta = Path.Combine(_vocesDir, id);
        if (Directory.Exists(carpeta)) Directory.Delete(carpeta, true);
        return Task.CompletedTask;
    }

    public async Task<byte[]> GetVoiceAudioAsync(string id)
    {
        var audioPath = Path.Combine(_vocesDir, id, "referencia.wav");
        if (!File.Exists(audioPath)) throw new FileNotFoundException("Audio no encontrado", audioPath);
        return await File.ReadAllBytesAsync(audioPath);
    }

    private static double GetDuration(string rutaWav)
    {
        try
        {
            using var reader = new NAudio.Wave.WaveFileReader(rutaWav);
            return reader.TotalTime.TotalSeconds;
        }
        catch { return 0; }
    }
}
