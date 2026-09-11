using CommunityToolkit.Mvvm.Input;

namespace Novox.Core.Models;

public class VozInfo
{
    public string Id { get; set; } = "";
    public string Nombre { get; set; } = "";
    public string RutaAudio { get; set; } = "";
    public string? Transcripcion { get; set; }
    public double Duracion { get; set; }
    public DateTime Creada { get; set; }

    public RelayCommand<VozInfo?> PlayCommand { get; set; } = new(_ => { });
    public RelayCommand<VozInfo?> DeleteCommand { get; set; } = new(_ => { });
    public RelayCommand<VozInfo?> UseCommand { get; set; } = new(_ => { });
}
