using System.Text.Json.Serialization;
using CommunityToolkit.Mvvm.Input;

namespace Novox.Core.Models;

public class VozInfo
{
    public string Id { get; set; } = string.Empty;
    public string Nombre { get; set; } = string.Empty;
    public string RutaAudio { get; set; } = string.Empty;
    public string? Transcripcion { get; set; } = string.Empty;
    public double Duracion { get; set; }
    public DateTime Creada { get; set; }

    public string DuracionText => FormatearDuracion(Duracion);

    private static string FormatearDuracion(double duracion)
    {
        if (duracion < 0 || duracion > 3600 || double.IsNaN(duracion)) // Duración inválida
            return "00:00";

        var ts = TimeSpan.FromSeconds(Math.Round(duracion));
        var formattedDuration = ts.TotalHours >= 1
            ? $"{(int)ts.TotalHours:D2}:{ts.Minutes:D2}:{ts.Seconds:D2}"
            : $"{(int)ts.TotalMinutes:D2}:{ts.Seconds:D2}";

        return formattedDuration;
    }

    [JsonIgnore] public RelayCommand<VozInfo?> PlayCommand { get; set; } = new(_ => { });
    [JsonIgnore] public RelayCommand<VozInfo?> DeleteCommand { get; set; } = new(_ => { });
    [JsonIgnore] public RelayCommand<VozInfo?> UseCommand { get; set; } = new(_ => { });
}
