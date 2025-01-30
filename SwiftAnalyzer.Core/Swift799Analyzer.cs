using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SwiftAnalyzer.Core.Models;
using SwiftAnalyzer.Core.Services.ML;

namespace SwiftAnalyzer.Core
{
    public class Swift799Analyzer : IDisposable
    {
        private readonly DeBertaModel _model;
        private readonly bool _extractNarrativeOnly;

        public Swift799Analyzer(string modelPath, bool extractNarrativeOnly = false)
        {
            _model = new DeBertaModel(modelPath);
            _extractNarrativeOnly = extractNarrativeOnly;
        }

        public async Task<Swift799Message> AnalyzeMessage(string messageText)
        {
            // Prétraitement optionnel pour extraire la partie narrative
            var textToAnalyze = _extractNarrativeOnly 
                ? ExtractNarrativePart(messageText) 
                : messageText;

            var analysis = await _model.AnalyzeSwiftMessage(textToAnalyze);
            
            var message = new Swift799Message
            {
                OriginalText = messageText,
                MainTask = analysis.MainOperationType,
                HasConditionalActions = analysis.HasConditions
            };

            foreach (var subTask in analysis.SubTasks)
            {
                message.SubTasks.Add(new MessageTask
                {
                    Type = subTask.Type,
                    Text = subTask.Text,
                    Confidence = subTask.Confidence
                });
            }

            foreach (var timeConstraint in analysis.TimeConstraints)
            {
                if (timeConstraint.Score > 0.3)
                {
                    message.TimeConstraints.Add(new TimeConstraint
                    {
                        Type = timeConstraint.Label,
                        Confidence = timeConstraint.Score
                    });
                }
            }

            return message;
        }

        private string ExtractNarrativePart(string swift799Message)
        {
            try
            {
                // Patterns courants pour identifier la partie narrative
                var patterns = new[]
                {
                    @"77E:.*?}\r?\n(.*?)(?=-|}|$)",  // Cherche après 77E jusqu'à la fin ou un délimiteur
                    @"Narrative:(.*?)(?=-|}|$)",      // Cherche après "Narrative:" 
                    @"79:.*?\r?\n(.*?)(?=-|}|$)"     // Cherche après le champ 79
                };

                foreach (var pattern in patterns)
                {
                    var match = Regex.Match(swift799Message, pattern, RegexOptions.Singleline);
                    if (match.Success && match.Groups.Count > 1)
                    {
                        var narrative = match.Groups[1].Value.Trim();
                        if (!string.IsNullOrWhiteSpace(narrative))
                        {
                            // Nettoyer le texte
                            narrative = Regex.Replace(narrative, @"\r?\n", " ");  // Remplacer les sauts de ligne
                            narrative = Regex.Replace(narrative, @"\s+", " ");    // Normaliser les espaces
                            return narrative;
                        }
                    }
                }

                // Si aucun pattern ne correspond, retourner le message complet
                return swift799Message;
            }
            catch (Exception)
            {
                // En cas d'erreur, retourner le message complet
                return swift799Message;
            }
        }

        public void Dispose()
        {
            _model?.Dispose();
        }
    }
}
