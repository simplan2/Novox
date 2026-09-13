using System.IO;
using NAudio.Wave;
using Novox.Core.Models;

namespace Novox.Core.Services;

public interface IAudioCaptureService
{
    event Action<float[]>? OnLevelUpdated;
    event Action<MemoryStream>? OnRecordingComplete;
    bool IsRecording { get; }
    double Duration { get; }
    Task StartRecordingAsync(CancellationToken ct = default);
    void StopRecording();
    MemoryStream? GetRecording();
    void Dispose();
}

public class AudioCaptureService : IAudioCaptureService
{
    private WaveInEvent? _waveIn;
    private MemoryStream? _recordingStream;
    private MemoryStream? _lastRecording;
    private WaveFileWriter? _writer;
    private bool _isRecording;
    private readonly object _lock = new();

    public event Action<float[]>? OnLevelUpdated;
    public event Action<MemoryStream>? OnRecordingComplete;
    public bool IsRecording => _isRecording;
    public double Duration { get; private set; }

    public async Task StartRecordingAsync(CancellationToken ct = default)
    {
        lock (_lock)
        {
            if (_isRecording) throw new InvalidOperationException("Ya se está grabando.");
            _lastRecording?.Dispose();
            _lastRecording = null;
            _recordingStream = new MemoryStream();
            _waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(44100, 16, 1),
                BufferMilliseconds = 50
            };
            _waveIn.DataAvailable += OnDataAvailable;
            _waveIn.RecordingStopped += OnRecordingStopped;
            _writer = new WaveFileWriter(_recordingStream, _waveIn.WaveFormat);
            _waveIn.StartRecording();
            _isRecording = true;
        }
    }

    public void StopRecording()
    {
        lock (_lock)
        {
            if (!_isRecording) return;
            _waveIn?.StopRecording();
        }
    }

    public MemoryStream? GetRecording()
    {
        lock (_lock)
        {
            // Vale tanto a mitad de grabación como DESPUÉS de detenerla
            // (antes se devolvía null siempre tras StopRecording).
            var src = _recordingStream ?? _lastRecording;
            if (src is null) return null;
            src.Position = 0;
            var copia = new MemoryStream();
            src.CopyTo(copia);
            copia.Position = 0;
            return copia;
        }
    }

    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        _writer?.Write(e.Buffer, 0, e.BytesRecorded);
        _writer?.Flush();

        var samples = new float[e.BytesRecorded / 2];
        Buffer.BlockCopy(e.Buffer, 0, samples, 0, e.BytesRecorded);
        var levels = new float[Math.Min(64, samples.Length)];
        for (int i = 0; i < levels.Length; i++)
            levels[i] = Math.Abs(samples[i * (samples.Length / levels.Length)]);
        OnLevelUpdated?.Invoke(levels);
    }

    private void OnRecordingStopped(object? sender, StoppedEventArgs e)
    {
        lock (_lock)
        {
            _isRecording = false;
            _recordingStream?.Position = 0;
            Duration = _recordingStream?.Length > 0 ? (double)_recordingStream.Length / (44100 * 2) : 0;
            var stream = _recordingStream;
            _lastRecording = stream; // conservar para Guardar voz
            _recordingStream = null;
            _writer?.Dispose();
            _writer = null;
            _waveIn?.Dispose();
            _waveIn = null;
            OnRecordingComplete?.Invoke(stream!);
        }
    }

    public void Dispose()
    {
        _waveIn?.Dispose();
        _writer?.Dispose();
        _recordingStream?.Dispose();
    }
}
