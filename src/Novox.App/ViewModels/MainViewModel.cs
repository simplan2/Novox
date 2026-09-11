using System.Collections.ObjectModel;
using System.Diagnostics;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Novox.Core.Models;
using Novox.Core.Services;

namespace Novox.App.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly AppConfig _config = new();
    private readonly ITtsService _tts;
    private readonly IAudioCaptureService _audioCapture;
    private readonly IVoiceLibraryService _voiceLib;
    private readonly IModelDownloadService _modelos;
    private CancellationTokenSource? _ctsGen;

    [ObservableProperty] private string _estadoTexto = "Inicializando...";
    [ObservableProperty] private string _infoSistema = "";
    [ObservableProperty] private string _infoMotor = "llama.cpp: comprobando…";
    [ObservableProperty] private string _texto = "";
    [ObservableProperty] private string _idiomaSeleccionado = "auto";
    [ObservableProperty] private string _dispositivoSeleccionado = "auto";
    public ObservableCollection<string> Idiomas { get; set; } = new(["auto", "es", "en", "zh", "de", "it", "pt", "ja", "ko", "fr", "ru"]);
    public ObservableCollection<string> Dispositivos { get; set; } = new(["auto", "CPU", "Vulkan0", "Vulkan1"]);
    public ObservableCollection<string> Motores { get; set; } = new(["auto", "llama", "onnx", "local"]);
    [ObservableProperty] private string _motorSeleccionado = "auto";

    [ObservableProperty] private string _progresoTexto = "";
    [ObservableProperty] private double _progresoPorcentaje;
    [ObservableProperty] private bool _puedeGenerar = true;
    [ObservableProperty] private bool _mostrarCancelar = false;
    [ObservableProperty] private string _resultadoInfo = "";
    [ObservableProperty] private string _cronometro = "0.0 s";
    [ObservableProperty] private string _grabarTexto = "● Grabar";
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _nombreVoz = "";
    [ObservableProperty] private string _transcripcionVoz = "";
    [ObservableProperty] private string _archivoSeleccionado = "";
    [ObservableProperty] private string _grabarColor = "#3fb950";

    [ObservableProperty] private ObservableCollection<VozInfo> _voces = new();
    [ObservableProperty] private VozInfo? _vozSeleccionada;
    [ObservableProperty] private bool _hayVoces;
    [ObservableProperty] private bool _hayHistorial;
    [ObservableProperty] private ObservableCollection<GeneracionResultado> _generaciones = new();
    [ObservableProperty] private ObservableCollection<double> _levelBars = new(Enumerable.Repeat(0.0, 32));
    [ObservableProperty] private ObservableCollection<string> _registro = new();

    [ObservableProperty] private int _selectedTabIndex;
    public ObservableCollection<TabItemInfo> TabItems { get; set; } = new();

    [ObservableProperty] private bool _tab1Visible = true;
    [ObservableProperty] private bool _tab2Visible = false;
    [ObservableProperty] private bool _tab3Visible = false;
    [ObservableProperty] private bool _tab4Visible = false;
    [ObservableProperty] private string _tab1Color = "#4f8cff";
    [ObservableProperty] private string _tab2Color = "#1e2530";
    [ObservableProperty] private string _tab3Color = "#1e2530";
    [ObservableProperty] private string _tab4Color = "#1e2530";

    // Modelos GGUF
    public ObservableCollection<ModelCatalog.Entrada> ModeloOpciones { get; } = new(ModelCatalog.Modelos.Values);
    public ObservableCollection<ModelCatalog.Entrada> MmprojOpciones { get; } = new(ModelCatalog.Mmproj.Values);
    [ObservableProperty] private ModelCatalog.Entrada? _modeloSeleccionado;
    [ObservableProperty] private ModelCatalog.Entrada? _mmprojSeleccionado;
    [ObservableProperty] private string _modeloEstado = "Comprobando modelos…";
    [ObservableProperty] private string _modeloProgresoTexto = "";
    [ObservableProperty] private double _modeloProgreso;
    [ObservableProperty] private bool _descargandoModelo;

    // Ajustes avanzados (paridad con Clonar-voz)
    [ObservableProperty] private double _temperatura = 0.8;
    [ObservableProperty] private double _topP = 0.95;
    [ObservableProperty] private int _topK = 40;
    [ObservableProperty] private int _semilla = -1;
    [ObservableProperty] private int _maxFrames = 1200;
    [ObservableProperty] private int _hilos;
    [ObservableProperty] private int _caracteresBloque = 280;
    [ObservableProperty] private int _pausaMs = 150;

    public MainViewModel()
    {
        _tts = new TtsService(_config);
        _audioCapture = new AudioCaptureService();
        _voiceLib = new VoiceLibraryService(_config);
        _modelos = new ModelDownloadService(_config);
        _modelos.OnProgress += m => Dispatcher.UIThread.Post(() => { ModeloProgresoTexto = m; AgregarLog("[modelo] " + m); });
        _modelos.OnProgressPercent += p => Dispatcher.UIThread.Post(() => ModeloProgreso = p);
        _modelos.OnComplete += () => Dispatcher.UIThread.Post(() =>
        {
            DescargandoModelo = false;
            ModeloProgreso = 100;
            ModeloProgresoTexto = "✓ Modelos listos.";
            RefrescarEstadoModelo();
        });
        _modelos.OnError += e => Dispatcher.UIThread.Post(() =>
        {
            DescargandoModelo = false;
            ModeloProgresoTexto = $"✕ {e}";
        });

        ModeloSeleccionado = ModeloOpciones.FirstOrDefault(o => o.Clave == ModelCatalog.PorDefectoModelo);
        MmprojSeleccionado = MmprojOpciones.FirstOrDefault(o => o.Clave == ModelCatalog.PorDefectoMmproj);

        TabItems.Add(new TabItemInfo { Title = "Voz de referencia" });
        TabItems.Add(new TabItemInfo { Title = "Texto a sintetizar" });
        TabItems.Add(new TabItemInfo { Title = "Audios generados" });
        TabItems.Add(new TabItemInfo { Title = "Modelos" });

        _ = InicializarAsync();
    }

    private void AgregarLog(string linea)
    {
        Registro.Add($"[{DateTime.Now:HH:mm:ss}] {linea}");
        while (Registro.Count > 200) Registro.RemoveAt(0);
    }

    [RelayCommand]
    private async Task InicializarAsync()
    {
        try
        {
            EstadoTexto = "Cargando motor de voz...";
            await _tts.InitializeAsync(new Progress<string>(msg => EstadoTexto = msg));
            await CargarVocesAsync();
            CargarHistorial();
            RefrescarEstadoLlama();
            RefrescarEstadoModelo();
            EstadoTexto = "Listo.";
            InfoSistema = $"{Environment.ProcessorCount} cores";
        }
        catch (Exception ex)
        {
            EstadoTexto = $"Error: {ex.Message}";
            InfoSistema = "Error";
        }
    }

    [RelayCommand]
    private async Task CargarVocesAsync()
    {
        var sel = VozSeleccionada?.Id;
        var voces = await _voiceLib.GetVoicesAsync();
        foreach (var voz in voces)
        {
            voz.PlayCommand = new RelayCommand<VozInfo?>(v => PlayVoz(v));
            voz.DeleteCommand = new RelayCommand<VozInfo?>(v => DeleteVoz(v));
            voz.UseCommand = new RelayCommand<VozInfo?>(v => UseVoz(v));
        }
        Voces = new ObservableCollection<VozInfo>(voces);
        HayVoces = Voces.Count > 0;
        VozSeleccionada = Voces.FirstOrDefault(v => v.Id == sel);
    }

    private void CargarHistorial()
    {
        try
        {
            var dir = string.IsNullOrEmpty(_config.OutputsDirectory)
                ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "salidas")
                : _config.OutputsDirectory;
            if (!Directory.Exists(dir)) return;
            var lista = new List<GeneracionResultado>();
            foreach (var f in Directory.GetFiles(dir, "*.wav").OrderByDescending(f => File.GetCreationTime(f)).Take(80))
            {
                var r = new GeneracionResultado
                {
                    Id = Path.GetFileNameWithoutExtension(f),
                    RutaArchivo = f,
                    NombreArchivo = Path.GetFileName(f),
                    Duracion = Math.Round(WavUtils.Duracion(f), 2),
                    SegundosGeneracion = 0,
                    Dispositivo = "",
                    Idioma = "",
                    Texto = Path.GetFileName(f),
                    Fecha = File.GetCreationTime(f),
                    VozId = null
                };
                r.PlayHistCommand = new RelayCommand<GeneracionResultado?>(_ => PlayHist(r));
                r.DescargarCommand = new RelayCommand<GeneracionResultado?>(_ => Descargar(r));
                r.DeleteHistCommand = new RelayCommand<GeneracionResultado?>(_ => DeleteHist(r));
                lista.Add(r);
            }
            Generaciones = new ObservableCollection<GeneracionResultado>(lista);
            HayHistorial = Generaciones.Count > 0;
        }
        catch { }
    }

    [RelayCommand]
    private async Task GrabarAsync()
    {
        if (IsRecording)
        {
            _audioCapture.StopRecording();
            return;
        }

        IsRecording = true;
        GrabarTexto = "■ Detener";
        Cronometro = "0.0 s";

        var sw = System.Diagnostics.Stopwatch.StartNew();
        _audioCapture.OnLevelUpdated += levels =>
        {
            var avg = levels.Length > 0 ? levels.Average() : 0;
            var bars = new ObservableCollection<double>(Enumerable.Repeat(0.0, 32));
            for (int i = 0; i < bars.Count; i++)
                bars[i] = Math.Min(3.0, avg * 30);
            LevelBars = bars;
        };

        _audioCapture.OnRecordingComplete += _ =>
        {
            IsRecording = false;
            GrabarTexto = "● Grabar";
            LevelBars = new ObservableCollection<double>(Enumerable.Repeat(0.0, 32));
            Cronometro = $"{sw.Elapsed.TotalSeconds:F1} s";
        };

        await _audioCapture.StartRecordingAsync();
    }

    [RelayCommand]
    private async Task GuardarVozAsync()
    {
        MemoryStream? stream = _audioCapture.GetRecording();
        if (stream is null && File.Exists(ArchivoSeleccionado))
        {
            var bytes = await File.ReadAllBytesAsync(ArchivoSeleccionado);
            stream = new MemoryStream(bytes);
        }
        if (stream is null) { EstadoTexto = "Nada que guardar: graba o indica una ruta de audio."; return; }
        if (string.IsNullOrWhiteSpace(NombreVoz)) NombreVoz = Path.GetFileNameWithoutExtension(ArchivoSeleccionado.Trim()) is { Length: > 0 } n ? n : "Voz sin nombre";

        await _voiceLib.AddVoiceAsync(NombreVoz, TranscripcionVoz, stream);
        await CargarVocesAsync();
        NombreVoz = "";
        TranscripcionVoz = "";
        ArchivoSeleccionado = "";
        EstadoTexto = "Voz guardada en la biblioteca.";
    }

    [RelayCommand]
    private async Task SubirArchivoAsync()
    {
        // Sin diálogo de archivos en este paso: pega la ruta en el campo y pulsa de nuevo.
        var ruta = ArchivoSeleccionado.Trim().Trim('"');
        if (!File.Exists(ruta))
        {
            EstadoTexto = "Pega la ruta del audio en el campo (p. ej. C:\\audios\\voz.wav) y pulsa «Importar».";
            return;
        }
        if (new FileInfo(ruta).Length > 60L * 1024 * 1024)
        {
            EstadoTexto = "El audio no puede pasar de 60 MB. Con 10-15 s basta.";
            return;
        }
        var nombre = string.IsNullOrWhiteSpace(NombreVoz) ? Path.GetFileNameWithoutExtension(ruta) : NombreVoz;
        var bytes = await File.ReadAllBytesAsync(ruta);
        await _voiceLib.AddVoiceAsync(nombre, TranscripcionVoz, new MemoryStream(bytes));
        await CargarVocesAsync();
        NombreVoz = "";
        TranscripcionVoz = "";
        EstadoTexto = $"Voz «{nombre}» importada.";
    }

    [RelayCommand]
    private async Task GenerarAsync()
    {
        if (string.IsNullOrWhiteSpace(Texto)) return;
        _ctsGen = new CancellationTokenSource();
        var ct = _ctsGen.Token;

        // Sincroniza UI → config (paridad Clonar-voz)
        _config.Temp = Temperatura;
        _config.TopP = TopP;
        _config.TopK = TopK;
        _config.Semilla = Semilla;
        _config.MaxFrames = MaxFrames;
        _config.Hilos = Hilos;
        _config.CaracteresPorBloque = CaracteresBloque;
        _config.PausaMs = PausaMs;
        _config.Dispositivo = DispositivoSeleccionado;
        _config.Idioma = IdiomaSeleccionado;
        _config.Motor = MotorSeleccionado;

        PuedeGenerar = false;
        MostrarCancelar = true;
        ProgresoTexto = "Iniciando generación...";
        ProgresoPorcentaje = 0;
        Registro.Clear();

        var bloques = TextChunker.Trocear(Texto, CaracteresBloque);
        AgregarLog($"Generando {bloques.Count} bloque(s) · motor={MotorSeleccionado} · voz={(VozSeleccionada?.Nombre ?? "defecto")} · disp={DispositivoSeleccionado} · temp={Temperatura} top_p={TopP} top_k={TopK} semilla={Semilla}");

        var progreso = new Progress<ProgresoInfo>(p =>
        {
            if (p.Estado == "bloque")
            {
                ProgresoTexto = $"Bloque {p.BloqueActual}/{p.TotalBloques}";
                ProgresoPorcentaje = p.Porcentaje;
            }
            else if (p.Estado == "log" && p.Mensaje is not null)
                AgregarLog(p.Mensaje);
            else if (p.Estado == "fin")
                ProgresoPorcentaje = 100;
        });

        try
        {
            var resultado = await _tts.SynthesizeAsync(
                Texto, VozSeleccionada?.RutaAudio, VozSeleccionada?.Id ?? "defecto",
                IdiomaSeleccionado, progreso, ct);
            ProgresoTexto = $"✓ Listo en {resultado.SegundosGeneracion:F1}s · {resultado.Duracion:F1}s";
            ResultadoInfo = $"✓ {resultado.NombreArchivo} · {resultado.Duracion:F1}s · {resultado.Dispositivo}";
            AgregarLog($"OK: {resultado.NombreArchivo} ({resultado.Duracion:F1}s)");
            resultado.PlayHistCommand = new RelayCommand<GeneracionResultado?>(_ => PlayHist(resultado));
            resultado.DescargarCommand = new RelayCommand<GeneracionResultado?>(_ => Descargar(resultado));
            resultado.DeleteHistCommand = new RelayCommand<GeneracionResultado?>(_ => DeleteHist(resultado));
            Generaciones.Insert(0, resultado);
            HayHistorial = true;
        }
        catch (OperationCanceledException)
        {
            ProgresoTexto = "■ Cancelada.";
            AgregarLog("Generación cancelada por el usuario.");
        }
        catch (Exception ex)
        {
            ProgresoTexto = $"✕ Error: {ex.Message}";
            AgregarLog($"ERROR: {ex.Message}");
        }
        finally
        {
            PuedeGenerar = true;
            MostrarCancelar = false;
            _ctsGen = null;
        }
    }

    [RelayCommand]
    private void CancelarGeneracion()
    {
        _ctsGen?.Cancel();
        ProgresoTexto = "Cancelando...";
    }

    private static void AbrirConReproductor(string ruta)
    {
        if (!File.Exists(ruta)) return;
        Process.Start(new ProcessStartInfo { FileName = ruta, UseShellExecute = true });
    }

    [RelayCommand]
    private void PlayHist(GeneracionResultado? gen)
    {
        if (gen is not null) AbrirConReproductor(gen.RutaArchivo);
    }

    [RelayCommand]
    private void Descargar(GeneracionResultado? gen)
    {
        if (gen is null || !File.Exists(gen.RutaArchivo)) return;
        try
        {
            var dest = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), gen.NombreArchivo);
            File.Copy(gen.RutaArchivo, dest, true);
            ResultadoInfo = $"⬇ Copiado al escritorio: {dest}";
        }
        catch (Exception ex) { ResultadoInfo = $"✕ No se pudo copiar: {ex.Message}"; }
    }

    private void PlayVoz(VozInfo? voz)
    {
        if (voz is not null) AbrirConReproductor(voz.RutaAudio);
    }

    private void DeleteVoz(VozInfo? voz)
    {
        if (voz is null) return;
        try { _ = _voiceLib.DeleteVoiceAsync(voz.Id); } catch { }
        Voces.Remove(voz);
        HayVoces = Voces.Count > 0;
        if (VozSeleccionada?.Id == voz.Id) VozSeleccionada = null;
    }

    private void UseVoz(VozInfo? voz)
    {
        if (voz is null) return;
        VozSeleccionada = Voces.FirstOrDefault(v => v.Id == voz.Id) ?? voz;
        SwitchToTab2();
    }

    private void DeleteHist(GeneracionResultado? gen)
    {
        if (gen is null) return;
        try { if (File.Exists(gen.RutaArchivo)) File.Delete(gen.RutaArchivo); } catch { }
        Generaciones.Remove(gen);
        HayHistorial = Generaciones.Count > 0;
    }

    [RelayCommand]
    private void ToggleDevice() { }

    private void RefrescarEstadoLlama()
    {
        var e = LlamaCppSetup.ObtenerEstado(_config.Binario);
        InfoMotor = !e.Existe ? $"llama.cpp: no encontrado — {e.Sugerencia}"
            : !e.SoportaQwen3 ? $"llama.cpp: {e.Version} (sin --mmproj: {e.Sugerencia})"
            : $"llama.cpp: {e.Version} ✓";
    }

    private void RefrescarEstadoModelo()
    {
        ModeloEstado = _modelos.IsModelAvailable()
            ? "✓ Modelo listo: hay par modelo + mmproj en la carpeta de modelos."
            : "Falta el modelo: descarga el par modelo + mmproj (~1.5 GB) una sola vez.";
    }

    [RelayCommand]
    private async Task DescargarModeloAsync()
    {
        if (ModeloSeleccionado is null || MmprojSeleccionado is null) return;
        DescargandoModelo = true;
        ModeloProgreso = 0;
        ModeloProgresoTexto = "Iniciando descarga…";
        await _modelos.DownloadAsync(ModeloSeleccionado.Clave, MmprojSeleccionado.Clave);
    }

    [RelayCommand]
    private void CancelarDescargaModelo() => _modelos.Cancel();

    [RelayCommand]
    private async Task DescargarLlamaAsync()
    {
        try
        {
            PuedeGenerar = false;
            ProgresoTexto = "Descargando llama.cpp…";
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var binDir = Path.Combine(string.IsNullOrEmpty(_config.ModelDirectory)
                ? Path.Combine(baseDir, "modelos") : _config.ModelDirectory, "bin");
            var log = new Progress<string>(m => AgregarLog(m));
            var bin = await LlamaCppSetup.DescargarAsync(binDir, log);
            _config.Binario = bin;
            RefrescarEstadoLlama();
            ProgresoTexto = "✓ llama.cpp instalado.";
        }
        catch (Exception ex)
        {
            ProgresoTexto = $"✕ {ex.Message}";
            AgregarLog($"ERROR llama.cpp: {ex.Message}");
        }
        finally { PuedeGenerar = true; }
    }

    [RelayCommand]
    private void SwitchToTab1()
    {
        SelectedTabIndex = 0;
        Tab1Visible = true;
        Tab2Visible = false;
        Tab3Visible = false;
        Tab4Visible = false;
        Tab1Color = "#4f8cff";
        Tab2Color = "#1e2530";
        Tab3Color = "#1e2530";
        Tab4Color = "#1e2530";
    }

    [RelayCommand]
    private void SwitchToTab2()
    {
        SelectedTabIndex = 1;
        Tab1Visible = false;
        Tab2Visible = true;
        Tab3Visible = false;
        Tab4Visible = false;
        Tab1Color = "#1e2530";
        Tab2Color = "#4f8cff";
        Tab3Color = "#1e2530";
        Tab4Color = "#1e2530";
    }

    [RelayCommand]
    private void SwitchToTab3()
    {
        SelectedTabIndex = 2;
        Tab1Visible = false;
        Tab2Visible = false;
        Tab3Visible = true;
        Tab4Visible = false;
        Tab1Color = "#1e2530";
        Tab2Color = "#1e2530";
        Tab3Color = "#4f8cff";
        Tab4Color = "#1e2530";
    }

    [RelayCommand]
    private void SwitchToTab4()
    {
        SelectedTabIndex = 3;
        Tab1Visible = false;
        Tab2Visible = false;
        Tab3Visible = false;
        Tab4Visible = true;
        Tab1Color = "#1e2530";
        Tab2Color = "#1e2530";
        Tab3Color = "#1e2530";
        Tab4Color = "#4f8cff";
        RefrescarEstadoModelo();
    }
}

public class TabItemInfo
{
    public string Title { get; set; } = "";
}
