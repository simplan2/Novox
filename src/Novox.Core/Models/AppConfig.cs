using System.Text.Json;

namespace Novox.Core.Models;

/// <summary>
/// Paridad con CONFIG_POR_DEFECTO de Clonar-voz (app.py).
/// </summary>
public class AppConfig
{
    public string Binario { get; set; } = "";
    public string Modelo { get; set; } = "";
    public string Mmproj { get; set; } = "";
    public string Dispositivo { get; set; } = "auto";
    public string Motor { get; set; } = "auto";
    public int CapasGpu { get; set; } = 99;
    public int Hilos { get; set; } = 0;
    public string Idioma { get; set; } = "es";
    public int TopK { get; set; } = 40;
    public double TopP { get; set; } = 0.95;
    public double Temp { get; set; } = 0.8;
    public int Semilla { get; set; } = -1;
    public int MaxFrames { get; set; } = 1200;
    public int CaracteresPorBloque { get; set; } = 280;
    public int PausaMs { get; set; } = 150;

    // Rutas de trabajo (no existen en Clonar-voz como config, pero Novox las necesita)
    public string ModelPath { get; set; } = "";
    public string ModelVariant { get; set; } = "Qwen17B";
    public string ExecutionProvider { get; set; } = "Cpu";
    public int GpuDeviceId { get; set; } = 0;
    public string DefaultSpeaker { get; set; } = "ryan";
    public string DefaultLanguage { get; set; } = "auto";
    public int MaxConcurrency { get; set; } = 1;
    public string VoicesDirectory { get; set; } = "";
    public string OutputsDirectory { get; set; } = "";
    public string ModelDirectory { get; set; } = "";

    public static AppConfig Load(string ruta)
    {
        var cfg = new AppConfig();
        try
        {
            if (!File.Exists(ruta)) return cfg;
            var json = File.ReadAllText(ruta);
            var doc = JsonDocument.Parse(json).RootElement;
            foreach (var prop in typeof(AppConfig).GetProperties())
            {
                if (!doc.TryGetProperty(PropName(prop.Name), out var el)) continue;
                try { prop.SetValue(cfg, el.Deserialize(prop.PropertyType)); } catch { }
            }
        }
        catch { }
        return cfg;
    }

    public void Save(string ruta)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var prop in typeof(AppConfig).GetProperties())
            dict[PropName(prop.Name)] = prop.GetValue(this);
        File.WriteAllText(ruta, JsonSerializer.Serialize(dict, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static string PropName(string name) => char.ToLowerInvariant(name[0]) + name[1..];
}
