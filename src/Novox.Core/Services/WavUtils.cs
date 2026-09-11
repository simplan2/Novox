namespace Novox.Core.Services;

/// <summary>
/// Utilidades WAV: port de duracion_wav() y unir_wavs() de Clonar-voz.
/// Todo en PCM 16-bit mono 16 kHz (formato que espera Qwen3-TTS).
/// Sin dependencias nativas: lee/escribe el header RIFF a mano.
/// </summary>
public static class WavUtils
{
    public const int SampleRate = 16000;

    public static double Duracion(string ruta)
    {
        try
        {
            using var fs = File.OpenRead(ruta);
            using var br = new BinaryReader(fs);
            return ReadDuration(br);
        }
        catch { return 0; }
    }

    public static void EscribirSilencio(string destino, double segundos)
    {
        var muestras = Math.Max(1, (int)(SampleRate * segundos));
        var datos = new short[muestras]; // ceros = silencio
        EscribirPcm16Mono(destino, datos, SampleRate);
    }

    /// <summary>Concatena WAVs del mismo formato intercalando silencio (pausa_ms).</summary>
    public static void Unir(string[] partes, string destino, int pausaMs = 150)
    {
        var validas = partes.Where(p => File.Exists(p) && new FileInfo(p).Length > 44).ToArray();
        if (validas.Length == 0) throw new InvalidOperationException("El modelo no generó ningún audio.");
        if (validas.Length == 1) { File.Copy(validas[0], destino, true); return; }

        var todo = new List<short>();
        var silencioMuestras = (int)(SampleRate * pausaMs / 1000.0);
        for (var i = 0; i < validas.Length; i++)
        {
            todo.AddRange(LeerPcm16Mono(validas[i]));
            if (i < validas.Length - 1)
                for (var s = 0; s < silencioMuestras; s++) todo.Add(0);
        }
        EscribirPcm16Mono(destino, todo.ToArray(), SampleRate);
    }

    public static short[] LeerPcm16Mono(string ruta)
    {
        using var fs = File.OpenRead(ruta);
        using var br = new BinaryReader(fs);
        // RIFF....WAVEfmt + data (tolera chunks extra buscando "data")
        br.ReadBytes(12);
        while (fs.Position < fs.Length - 8)
        {
            var id = new string(br.ReadChars(4));
            var tam = br.ReadInt32();
            if (id == "data")
            {
                var bytes = br.ReadBytes(tam);
                var out_ = new short[bytes.Length / 2];
                Buffer.BlockCopy(bytes, 0, out_, 0, bytes.Length);
                return out_;
            }
            fs.Seek(tam, SeekOrigin.Current);
        }
        return Array.Empty<short>();
    }

    public static void EscribirPcm16Mono(string destino, short[] muestras, int hz)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(destino))!);
        using var fs = File.Create(destino);
        using var bw = new BinaryWriter(fs);
        var bytesDatos = muestras.Length * 2;
        bw.Write("RIFF".ToCharArray());
        bw.Write(36 + bytesDatos);
        bw.Write("WAVE".ToCharArray());
        bw.Write("fmt ".ToCharArray());
        bw.Write(16);
        bw.Write((short)1);
        bw.Write((short)1);
        bw.Write(hz);
        bw.Write(hz * 2);
        bw.Write((short)2);
        bw.Write((short)16);
        bw.Write("data".ToCharArray());
        bw.Write(bytesDatos);
        foreach (var m in muestras) bw.Write(m);
    }

    private static double ReadDuration(BinaryReader br)
    {
        var fs = br.BaseStream;
        br.ReadBytes(12);
        int hz = SampleRate, canales = 1, bits = 16;
        long datosBytes = 0;
        while (fs.Position < fs.Length - 8)
        {
            var id = new string(br.ReadChars(4));
            var tam = br.ReadInt32();
            if (id == "fmt ")
            {
                br.ReadInt16(); canales = br.ReadInt16(); hz = br.ReadInt32();
                br.ReadInt32(); br.ReadInt16(); bits = br.ReadInt16();
                if (tam > 16) fs.Seek(tam - 16, SeekOrigin.Current);
            }
            else if (id == "data") { datosBytes = tam; break; }
            else fs.Seek(tam, SeekOrigin.Current);
        }
        if (hz <= 0) return 0;
        return datosBytes / (double)(hz * canales * (bits / 8));
    }
}
