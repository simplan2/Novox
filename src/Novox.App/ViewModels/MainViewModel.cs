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
    private readonly AppConfig _config;
    private readonly string _configPath;
    private readonly ITtsService _tts;
    private readonly IAudioCaptureService _audioCapture;
    private readonly IVoiceLibraryService _voiceLib;
    private readonly IModelDownloadService _modelos;
    private CancellationTokenSource? _ctsGen;
    private readonly System.Diagnostics.Stopwatch _swGrabacion = new();
    private readonly DispatcherTimer _timerGrabacion = new() { Interval = TimeSpan.FromMilliseconds(100) };
    private double _progresoObjetivo;
    private readonly DispatcherTimer _timerProgreso = new() { Interval = TimeSpan.FromMilliseconds(200) };
    private double _currentLevelBarWidth = 0; // Guarda el valor anterior para suavizar

    [ObservableProperty] private string _estadoTexto = "Inicializando...";
    [ObservableProperty] private string _infoSistema = "";
    [ObservableProperty] private string _infoMotor = "llama.cpp: comprobando…";
    [ObservableProperty] private string _texto = "";
    [ObservableProperty] private string _idiomaSeleccionado = "es";
    [ObservableProperty] private string _dispositivoSeleccionado = "auto";
    public ObservableCollection<string> Idiomas { get; set; } = new(["es", "en", "zh", "de", "it", "pt", "ja", "ko", "fr", "ru"]);
    [ObservableProperty] private ObservableCollection<string> _dispositivos = new(["auto", "CPU"]);
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
    [ObservableProperty] private string _nombreArchivo = "";
    [ObservableProperty] private string _transcripcionArchivo = "";
    [ObservableProperty] private string _archivoSeleccionado = "";
    [ObservableProperty] private string _grabarColor = "#f85149"; // rojo: estándar "grabar"

    [ObservableProperty] private ObservableCollection<VozInfo> _voces = new();
    [ObservableProperty] private VozInfo? _vozSeleccionada;
    [ObservableProperty] private bool _hayVoces;
    [ObservableProperty] private bool _hayHistorial;

    /// <summary>
    /// Opciones del desplegable VOZ: la voz de serie del modelo (única voz
    /// integrada del backend llama: Qwen3-TTS Base no tiene catálogo de
    /// presets, solo referencia o defecto) + la biblioteca propia.
    /// </summary>
    public const string VozDefectoId = "__defecto__";
    private static VozInfo VozDefecto() => new()
    {
        Id = VozDefectoId,
        Nombre = "Voz del modelo (por defecto)",
        RutaAudio = "",
        Transcripcion = "Voz de serie del modelo, sin clonar.",
        Duracion = 0,
        Creada = DateTime.Now,
    };
    [ObservableProperty] private ObservableCollection<VozInfo> _vocesParaElegir = new([VozDefecto()]);
    [ObservableProperty] private ObservableCollection<GeneracionResultado> _generaciones = new();
    //[ObservableProperty] private ObservableCollection<double> _levelBars = new(Enumerable.Repeat(2.0, 32));
    private double _audioLevelWidth;
    public double AudioLevelWidth
    {
        get => _audioLevelWidth;
        set
        {
            if (_audioLevelWidth != value)
            {
                _audioLevelWidth = value;
                OnPropertyChanged(); // ¡Esto es lo que avisa a la UI que debe redibujarse!
            }
        }
    }
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

    // Ajustes avanzados (paridad con Clonar-voz; por defecto = llama.cpp)
    [ObservableProperty] private double _temperatura = 0.8;
    [ObservableProperty] private double _topP = 0.95;
    [ObservableProperty] private int _topK = 40;
    [ObservableProperty] private int _semilla = -1;
    [ObservableProperty] private int _maxFrames = 1200;
    [ObservableProperty] private int _hilos;
    [ObservableProperty] private int _caracteresBloque = 280;
    [ObservableProperty] private int _pausaMs = 150;

    // Validación: fuera de rango → se recorta al mínimo/máximo.
    // Se publica vía Dispatcher: si se recorta dentro del propio ciclo del
    // binding, Avalonia ignora la notificación y la caja seguiría mostrando
    // el número gigante aunque el valor ya esté recortado.
    private void Recortar<T>(T valor, T recortado, Action<T> asignar) where T : IEquatable<T>
    {
        if (!recortado.Equals(valor))
            Dispatcher.UIThread.Post(() => asignar(recortado));
    }
    partial void OnTemperaturaChanged(double value) => Recortar(value, Math.Clamp(value, 0.1, 1.5), v => Temperatura = v);
    partial void OnTopPChanged(double value) => Recortar(value, Math.Clamp(value, 0.1, 1.0), v => TopP = v);
    partial void OnTopKChanged(int value) => Recortar(value, Math.Clamp(value, 0, 100), v => TopK = v);
    partial void OnSemillaChanged(int value) => Recortar(value, Math.Clamp(value, -1, 999999999), v => Semilla = v);
    partial void OnMaxFramesChanged(int value) => Recortar(value, Math.Clamp(value, 12, 4000), v => MaxFrames = v);
    partial void OnHilosChanged(int value) => Recortar(value, Math.Clamp(value, 0, 256), v => Hilos = v);
    partial void OnCaracteresBloqueChanged(int value) => Recortar(value, Math.Clamp(value, 50, 2000), v => CaracteresBloque = v);
    partial void OnPausaMsChanged(int value) => Recortar(value, Math.Clamp(value, 0, 2000), v => PausaMs = v);

    [RelayCommand]
    private void RestablecerAjustes()
    {
        var d = new AppConfig();
        Temperatura = d.Temp;
        TopP = d.TopP;
        TopK = d.TopK;
        Semilla = d.Semilla;
        MaxFrames = d.MaxFrames;
        Hilos = d.Hilos;
        CaracteresBloque = d.CaracteresPorBloque;
        PausaMs = d.PausaMs;
        AgregarLog("Ajustes restablecidos a los valores por defecto (llama.cpp).");
    }

    public MainViewModel()
    {
        _configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config.json");
        _config = AppConfig.Load(_configPath);
        // La config manda (paridad con config.json de Clonar-voz)
        Temperatura = _config.Temp;
        TopP = _config.TopP;
        TopK = _config.TopK;
        Semilla = _config.Semilla;
        MaxFrames = _config.MaxFrames;
        Hilos = _config.Hilos;
        CaracteresBloque = _config.CaracteresPorBloque;
        PausaMs = _config.PausaMs;
        if (Idiomas.Contains(_config.Idioma)) IdiomaSeleccionado = _config.Idioma;
        if (Dispositivos.Contains(_config.Dispositivo)) DispositivoSeleccionado = _config.Dispositivo;
        if (Motores.Contains(_config.Motor)) MotorSeleccionado = _config.Motor;

        _tts = new TtsService(_config);
        _audioCapture = new AudioCaptureService();
        _audioCapture.OnLevelUpdated += AlRecibirNivel;
        _audioCapture.OnRecordingComplete += AlTerminarGrabacion;
        _timerGrabacion.Tick += (_, _) => Cronometro = $"{_swGrabacion.Elapsed.TotalSeconds:F1} s";
        // Progreso continuo: entre eventos de bloque la barra avanza sola
        // hacia el objetivo (llama-tts no informa % dentro del bloque).
        _timerProgreso.Tick += (_, _) =>
        {
            var tope = _progresoObjetivo >= 100 ? 100 : _progresoObjetivo - 0.5;
            if (ProgresoPorcentaje < tope)
                ProgresoPorcentaje = Math.Min(tope, ProgresoPorcentaje + Math.Max(0.4, (tope - ProgresoPorcentaje) * 0.08));
        };
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
        var lista = new List<VozInfo> { VozDefecto() };
        lista.AddRange(voces);
        VocesParaElegir = new ObservableCollection<VozInfo>(lista);
        VozSeleccionada = VocesParaElegir.FirstOrDefault(v => v.Id == sel)
            ?? VocesParaElegir.FirstOrDefault(v => v.Id == VozSeleccionada?.Id)
            ?? VocesParaElegir[0];
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
                r.SaveVozCommand = new RelayCommand<GeneracionResultado?>(_ => GuardarEnVoces(r));
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

        try
        {
            IsRecording = true;
            GrabarTexto = "■ Detener";
            GrabarColor = "#d29922"; // ámbar: estándar "detener"
            _swGrabacion.Restart();
            Cronometro = "0.0 s";
            _timerGrabacion.Start();
            await _audioCapture.StartRecordingAsync();
        }
        catch (Exception ex)
        {
            _timerGrabacion.Stop();
            IsRecording = false;
            GrabarTexto = "● Grabar";
            GrabarColor = "#f85149";
            EstadoTexto = $"✕ No se pudo grabar (¿hay micrófono?): {ex.Message}";
        }
    }

    /// <summary>
    /// Medidor de nivel: NAudio avisa en hilo de fondo, así que se
    /// marshala al hilo de UI (si no, las barras no se mueven).
    /// </summary>
    private void AlRecibirNivel(float[] levels)
    {
        if (levels == null || levels.Length == 0) return;

        float maxSample = 0f;
        for (int i = 0; i < levels.Length; i++)
        {
            float val = Math.Abs(levels[i]);
            if (val > maxSample) maxSample = val;
        }

        // 1. Sensibilidad ajustada (puedes probar entre 3.0 y 6.0)
        double normalizedLevel = Math.Min(1.0, maxSample * 10.0);
        double maxWidth = 300;
        double targetWidth = normalizedLevel * maxWidth;

        // 2. Aplicar suavizado (Lerp): se mueve un 40% hacia el nuevo valor en cada frame/evento
        // Esto evita los saltos bruscos y da una sensación de "vúmetro analógico"
        _currentLevelBarWidth = _currentLevelBarWidth + (targetWidth - _currentLevelBarWidth) * 0.4;

        Dispatcher.UIThread.Post(() =>
        {
            AudioLevelWidth = _currentLevelBarWidth;
        });
    }

    private void AlTerminarGrabacion(MemoryStream _)
    {
        Dispatcher.UIThread.Post(() =>
        {
            _timerGrabacion.Stop();
            IsRecording = false;
            GrabarTexto = "● Grabar";
            GrabarColor = "#f85149";
            Cronometro = $"{_swGrabacion.Elapsed.TotalSeconds:F1} s";
            _currentLevelBarWidth = 0;
            AudioLevelWidth = 0;
        });
    }

    [RelayCommand]
    private async Task GuardarVozAsync()
    {
        try
        {
            MemoryStream? stream = _audioCapture.GetRecording();
            if (stream is null && File.Exists(ArchivoSeleccionado.Trim().Trim('"')))
            {
                var bytes = await File.ReadAllBytesAsync(ArchivoSeleccionado.Trim().Trim('"'));
                stream = new MemoryStream(bytes);
            }
            if (stream is null) { EstadoTexto = "Nada que guardar: graba o indica la ruta de un audio."; return; }
            if (string.IsNullOrWhiteSpace(NombreVoz)) NombreVoz = "Voz sin nombre";

            var voz = await _voiceLib.AddVoiceAsync(NombreVoz, TranscripcionVoz, stream);
            await CargarVocesAsync();
            VozSeleccionada = VocesParaElegir.FirstOrDefault(v => v.Id == voz.Id) ?? VozSeleccionada;
            NombreVoz = "";
            TranscripcionVoz = "";
            ArchivoSeleccionado = "";
            EstadoTexto = $"Voz «{voz.Nombre}» guardada ({voz.Duracion:F1}s). Ya puedes usarla en Sintetizar.";
        }
        catch (Exception ex) { EstadoTexto = $"✕ No se pudo guardar la voz: {ex.Message}"; }
    }

    [RelayCommand]
    private async Task SubirArchivoAsync()
    {
        try
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
            var nombre = string.IsNullOrWhiteSpace(NombreArchivo) ? Path.GetFileNameWithoutExtension(ruta) : NombreArchivo;
            var bytes = await File.ReadAllBytesAsync(ruta);
            var voz = await _voiceLib.AddVoiceAsync(nombre, TranscripcionArchivo, new MemoryStream(bytes));
            await CargarVocesAsync();
            VozSeleccionada = VocesParaElegir.FirstOrDefault(v => v.Id == voz.Id) ?? VozSeleccionada;
            NombreArchivo = "";
            TranscripcionArchivo = "";
            EstadoTexto = $"Voz «{nombre}» importada ({voz.Duracion:F1}s). Ya puedes usarla en Sintetizar.";
        }
        catch (Exception ex) { EstadoTexto = $"✕ No se pudo importar: {ex.Message}"; }
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
        try { _config.Save(_configPath); } catch { }

        PuedeGenerar = false;
        MostrarCancelar = true;
        ProgresoTexto = "Iniciando generación...";
        ProgresoPorcentaje = 0;
        _progresoObjetivo = 0;
        ResultadoInfo = "";
        Registro.Clear();
        _timerProgreso.Start();

        var bloques = TextChunker.Trocear(Texto, CaracteresBloque);
        var esDefecto = VozSeleccionada is null || VozSeleccionada.Id == VozDefectoId
            || string.IsNullOrEmpty(VozSeleccionada.RutaAudio) || !File.Exists(VozSeleccionada.RutaAudio);
        if (esDefecto)
            AgregarLog("Voz del modelo (por defecto, sin clonar). Para clonar elige una de Mis Voces.");
        else
            AgregarLog($"Clonando voz «{VozSeleccionada!.Nombre}» ({VozSeleccionada.Duracion:F1}s de referencia)…");

        var progreso = new Progress<ProgresoInfo>(p =>
        {
            if (p.Estado == "bloque")
            {
                var trozo = (p.TextoActual ?? "").Trim();
                if (trozo.Length > 60) trozo = trozo[..60].Trim() + "…";
                ProgresoTexto = $"Bloque {p.BloqueActual}/{p.TotalBloques} · {trozo}";
                _progresoObjetivo = p.TotalBloques == 0 ? 100 : 100.0 * p.BloqueActual / p.TotalBloques;
            }
            else if (p.Estado == "log" && p.Mensaje is not null)
                AgregarLog(p.Mensaje);
            else if (p.Estado == "fin")
            {
                _progresoObjetivo = 100;
                ProgresoPorcentaje = 100;
            }
        });

        try
        {
            var resultado = await _tts.SynthesizeAsync(
                Texto, esDefecto ? null : VozSeleccionada!.RutaAudio, esDefecto ? "defecto" : VozSeleccionada!.Id,
                IdiomaSeleccionado, progreso, ct);
            ProgresoTexto = $"✓ Listo en {resultado.SegundosGeneracion:F1}s · {resultado.Duracion:F1}s";
            ResultadoInfo = $"✓ {resultado.NombreArchivo} · {resultado.Duracion:F1}s · {resultado.Dispositivo} · voz={(esDefecto ? "modelo" : VozSeleccionada!.Nombre)}";
            AgregarLog($"OK: {resultado.NombreArchivo} ({resultado.Duracion:F1}s)");
            resultado.PlayHistCommand = new RelayCommand<GeneracionResultado?>(_ => PlayHist(resultado));
            resultado.DescargarCommand = new RelayCommand<GeneracionResultado?>(_ => Descargar(resultado));
            resultado.DeleteHistCommand = new RelayCommand<GeneracionResultado?>(_ => DeleteHist(resultado));
            resultado.SaveVozCommand = new RelayCommand<GeneracionResultado?>(_ => GuardarEnVoces(resultado));
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
            _timerProgreso.Stop();
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
        var quito = VocesParaElegir.FirstOrDefault(v => v.Id == voz.Id);
        if (quito is not null) VocesParaElegir.Remove(quito);
        if (VozSeleccionada?.Id == voz.Id) VozSeleccionada = VocesParaElegir[0];
    }

    private void UseVoz(VozInfo? voz)
    {
        if (voz is null) return;
        VozSeleccionada = VocesParaElegir.FirstOrDefault(v => v.Id == voz.Id) ?? voz;
        SwitchToTab2();
    }

    private void DeleteHist(GeneracionResultado? gen)
    {
        if (gen is null) return;
        try { if (File.Exists(gen.RutaArchivo)) File.Delete(gen.RutaArchivo); } catch { }
        Generaciones.Remove(gen);
        HayHistorial = Generaciones.Count > 0;
    }

    /// <summary>
    /// Convierte un audio generado en voz reutilizable de la biblioteca
    /// (su propio texto queda como transcripción, ideal para el parecido).
    /// </summary>
    private async void GuardarEnVoces(GeneracionResultado? gen)
    {
        if (gen is null || !File.Exists(gen.RutaArchivo)) return;
        try
        {
            var baseNombre = string.IsNullOrWhiteSpace(gen.Texto) ? gen.NombreArchivo : gen.Texto.Trim();
            if (baseNombre.Length > 32) baseNombre = baseNombre[..32].Trim() + "…";
            var bytes = await File.ReadAllBytesAsync(gen.RutaArchivo);
            var voz = await _voiceLib.AddVoiceAsync(baseNombre, gen.Texto, new MemoryStream(bytes));
            await CargarVocesAsync();
            VozSeleccionada = VocesParaElegir.FirstOrDefault(v => v.Id == voz.Id) ?? VozSeleccionada;
            EstadoTexto = $"Voz «{voz.Nombre}» creada desde el historial. Ya está seleccionada en Sintetizar.";
        }
        catch (Exception ex) { EstadoTexto = $"✕ No se pudo guardar en voces: {ex.Message}"; }
    }

    [RelayCommand]
    private void ToggleDevice() { }

    private void RefrescarEstadoLlama()
    {
        var e = LlamaCppSetup.ObtenerEstado(_config.Binario);
        InfoMotor = !e.Existe ? $"llama.cpp: no encontrado — {e.Sugerencia}"
            : !e.SoportaQwen3 ? $"llama.cpp: {e.Version} (sin --mmproj: {e.Sugerencia})"
            : $"llama.cpp: {e.Version} ✓";
        // Dispositivos reales (Vulkan0, CUDA0…) en vez de la lista quemada
        var ids = LlamaCppSetup.ListarDispositivos(e.Binario).Select(d => d.Id).ToList();
        var lista = new List<string> { "auto", "CPU" };
        lista.AddRange(ids.Where(id => !lista.Contains(id)));
        var sel = DispositivoSeleccionado;
        Dispositivos = new ObservableCollection<string>(lista);
        DispositivoSeleccionado = lista.Contains(sel) ? sel : "auto";
    }

    private void RefrescarEstadoModelo()
    {
        var dir = string.IsNullOrEmpty(_config.ModelDirectory)
            ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "modelos")
            : _config.ModelDirectory;
        if (_modelos.IsModelAvailable())
        {
            var archivos = Directory.Exists(dir)
                ? string.Join(", ", Directory.GetFiles(dir, "*.gguf").Select(Path.GetFileName))
                : "";
            ModeloEstado = $"✓ Modelo listo en:\n{dir}\n{archivos}";
        }
        else
        {
            ModeloEstado = $"Falta el modelo en:\n{dir}\nDescarga el par modelo + mmproj (~1.5 GB) una sola vez.";
        }
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
            try { _config.Save(_configPath); } catch { }
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
