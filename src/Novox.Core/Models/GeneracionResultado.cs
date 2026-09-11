using CommunityToolkit.Mvvm.Input;

namespace Novox.Core.Models;

public class GeneracionResultado
{
    public required string Id { get; init; }
    public required string RutaArchivo { get; init; }
    public required string NombreArchivo { get; init; }
    public double Duracion { get; init; }
    public double SegundosGeneracion { get; init; }
    public required string Dispositivo { get; init; }
    public required string Idioma { get; init; }
    public required string Texto { get; init; }
    public DateTime Fecha { get; init; }
    public string? VozId { get; init; }

    public string FechaFormateada => Fecha.ToString("dd/MM/yyyy HH:mm");
    public RelayCommand<GeneracionResultado?> PlayHistCommand { get; set; } = new(_ => { });
    public RelayCommand<GeneracionResultado?> DescargarCommand { get; set; } = new(_ => { });
    public RelayCommand<GeneracionResultado?> DeleteHistCommand { get; set; } = new(_ => { });
}
