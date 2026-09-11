using System.Text.RegularExpressions;

namespace Novox.Core.Services;

/// <summary>
/// Port de trocear_texto() de Clonar-voz (app.py): parte por frases para no
/// agotar el límite de frames del modelo.
/// </summary>
public static class TextChunker
{
    public static List<string> Trocear(string texto, int maximo)
    {
        texto = Regex.Replace(texto ?? "", @"\s+", " ").Trim();
        if (texto.Length == 0) return new();
        if (texto.Length <= maximo) return new() { texto };

        var bloques = new List<string>();
        var actual = "";
        var frases = Regex.Split(texto, @"(?<=[.!?…。！？;:])\s+");
        foreach (var f0 in frases)
        {
            var frase = f0;
            while (frase.Length > maximo)
            {
                var corte = Math.Max(frase.LastIndexOf(',', maximo), frase.LastIndexOf(' ', maximo));
                if (corte <= maximo / 2) corte = maximo;
                if (actual.Length > 0) { bloques.Add(actual.Trim()); actual = ""; }
                bloques.Add(frase[..corte].Trim());
                frase = frase[corte..].Trim();
            }
            if ((actual.Length + frase.Length + 1) <= maximo)
                actual = (actual + " " + frase).Trim();
            else
            {
                if (actual.Length > 0) bloques.Add(actual.Trim());
                actual = frase;
            }
        }
        if (actual.Length > 0) bloques.Add(actual.Trim());
        return bloques.Where(b => b.Length > 0).ToList();
    }
}
