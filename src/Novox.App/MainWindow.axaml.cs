using System.IO;
using System.Linq;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;

namespace Novox.App;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    /// <summary>
    /// Explorador nativo para elegir el audio de referencia
    /// (en vez de pegar la ruta a mano).
    /// </summary>
    private async void ExaminarArchivo_Click(object? sender, RoutedEventArgs e)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null) return;
        var archivos = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Elige un audio de referencia",
            AllowMultiple = false,
            FileTypeFilter =
            [
                new FilePickerFileType("Audio")
                {
                    Patterns = ["*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac", "*.wma", "*.webm"]
                },
                new FilePickerFileType("Todos") { Patterns = ["*.*"] }
            ]
        });
        var f = archivos.FirstOrDefault();
        if (f is null) return;
        var ruta = f.TryGetLocalPath();
        if (string.IsNullOrEmpty(ruta)) return;
        if (DataContext is ViewModels.MainViewModel vm)
        {
            vm.ArchivoSeleccionado = ruta;
            if (string.IsNullOrWhiteSpace(vm.NombreArchivo))
                vm.NombreArchivo = Path.GetFileNameWithoutExtension(ruta);
        }
    }
}
