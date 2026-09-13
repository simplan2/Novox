using NAudio.MediaFoundation;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Novox.Core.Services;

/// <summary>
/// Port de convertir_a_wav() de Clonar-voz: el modelo espera PCM mono de
/// 16 kHz. Si la referencia entra en otro formato (mp3, m4a, 44.1 kHz,
/// estéreo…), el embedding de voz sale basura y el clon "no se parece en
/// nada". Se normaliza al guardar cada voz, sin necesitar ffmpeg.
/// </summary>
public static class AudioNormalizer
{
    public const int Hz = 16000;

    public static MemoryStream Normalizar16kMono(Stream entrada)
    {
        entrada.Position = 0;
        var bytes = new byte[entrada.Length];
        var leidos = 0;
        while (leidos < bytes.Length)
        {
            var n = entrada.Read(bytes, leidos, bytes.Length - leidos);
            if (n == 0) break;
            leidos += n;
        }

        using var reader = AbrirLectura(bytes);
        ISampleProvider sp = reader.ToSampleProvider();
        if (reader.WaveFormat.Channels > 1)
            sp = new StereoToMonoSampleProvider(sp);
        if (reader.WaveFormat.SampleRate != Hz)
            sp = new WdlResamplingSampleProvider(sp, Hz);
        var salida = new MemoryStream();
        WaveFileWriter.WriteWavFileToStream(salida, sp.ToWaveProvider16());
        salida.Position = 0;
        return salida;
    }

    public static double DuracionSegundos(Stream wav)
    {
        try
        {
            wav.Position = 0;
            using var r = new WaveFileReader(wav);
            return r.TotalTime.TotalSeconds;
        }
        catch { return 0; }
    }

    private static WaveStream AbrirLectura(byte[] bytes)
    {
        // WAV directo (lo que graba el micrófono de Novox y la mayoría de subidas)
        try { return new WaveFileReader(new MemoryStream(bytes, writable: false)); }
        catch { }
        // MP3 puro sin depender de Media Foundation
        try { return new Mp3FileReader(new MemoryStream(bytes, writable: false)); }
        catch { }
        // Todo lo demás (m4a, wma, ogg…): Media Foundation de Windows
        var tmp = Path.Combine(Path.GetTempPath(), "novox", "ref-" + Guid.NewGuid().ToString("N") + ".bin");
        Directory.CreateDirectory(Path.GetDirectoryName(tmp)!);
        File.WriteAllBytes(tmp, bytes);
        try { return new AutoBorradoReader(new MediaFoundationReader(tmp), tmp); }
        catch
        {
            try { File.Delete(tmp); } catch { }
            throw new InvalidOperationException(
                "Formato de audio no reconocido. Convierte a WAV o MP3 (p. ej. con ffmpeg) e inténtalo de nuevo.");
        }
    }

    private sealed class AutoBorradoReader : WaveStream
    {
        private readonly WaveStream _inner;
        private readonly string _tmp;
        public AutoBorradoReader(WaveStream inner, string tmp) { _inner = inner; _tmp = tmp; }
        public override WaveFormat WaveFormat => _inner.WaveFormat;
        public override long Length => _inner.Length;
        public override long Position { get => _inner.Position; set => _inner.Position = value; }
        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        protected override void Dispose(bool disposing)
        {
            if (disposing) _inner.Dispose();
            try { File.Delete(_tmp); } catch { }
            base.Dispose(disposing);
        }
    }
}
