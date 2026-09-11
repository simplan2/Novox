namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Options for a text-to-speech synthesis request.
/// </summary>
public class TextToSpeechOptions
{
    /// <summary>Speaker/voice identifier (e.g., "ryan", "serena").</summary>
    public string? VoiceId { get; set; }

    /// <summary>Language code (e.g., "english", "auto").</summary>
    public string? Language { get; set; }

    /// <summary>Optional instruction for speech style (e.g., "speak with excitement").</summary>
    public string? Instruct { get; set; }

    /// <summary>Optional model identifier for the response metadata.</summary>
    public string? ModelId { get; set; }
}
