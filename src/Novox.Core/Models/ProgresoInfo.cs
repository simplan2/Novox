namespace Novox.Core.Models;

public class ProgresoInfo
{
    public required string Estado { get; init; }
    public string? Mensaje { get; init; }
    public int BloqueActual { get; init; }
    public int TotalBloques { get; init; }
    public double Porcentaje { get; init; }
    public string? TextoActual { get; init; }
}
