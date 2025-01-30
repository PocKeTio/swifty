using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SwiftAnalyzer.Core.Models;
using SwiftAnalyzer.Core.Services.ML;
using SubTask = SwiftAnalyzer.Core.Models.SubTask;

namespace SwiftAnalyzer.Core
{
    public class Swift799Analyzer : IDisposable
    {
        private readonly DeBertaModel _model;
        private readonly bool _extractNarrativeOnly;

        public Swift799Analyzer(string modelPath, string tokenizerPath, string vocabPath, bool extractNarrativeOnly = false)
        {
            _model = new DeBertaModel(modelPath, tokenizerPath, vocabPath);
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
                message.SubTasks.Add(new SubTask
                {
                    Type = subTask.Type,
                    Text = subTask.Text,
                    Confidence = subTask.Confidence,
                    RelatedTasks = subTask.RelatedTasks
                });
            }

            foreach (var timeConstraint in analysis.TimeConstraints)
            {
                if (timeConstraint.Item2 > 0.3)
                {
                    message.TimeConstraints.Add(new TimeConstraint
                    {
                        Type = timeConstraint.Item1,
                        Confidence = timeConstraint.Item2
                    });
                }
            }

            return message;
        }

        private string ExtractNarrativePart(string messageText)
        {
            // Recherche de la partie narrative après les en-têtes et codes
            var narrativeMatch = Regex.Match(messageText, @"(?:^|\n)\s*77E:?\s*(.+?)(?:\n|$)", RegexOptions.Singleline);
            if (narrativeMatch.Success)
            {
                return narrativeMatch.Groups[1].Value.Trim();
            }

            // Si pas de champ 77E, chercher après le dernier champ numéroté
            var lastFieldMatch = Regex.Match(messageText, @"(?:^|\n)\s*\d{2}[A-Z]:?\s*(.+?)(?:\n|$)", RegexOptions.Singleline | RegexOptions.RightToLeft);
            if (lastFieldMatch.Success)
            {
                return lastFieldMatch.Groups[1].Value.Trim();
            }

            // Si aucun format reconnu, retourner le texte complet
            return messageText.Trim();
        }

        public void Dispose()
        {
            _model?.Dispose();
        }
    }
}
