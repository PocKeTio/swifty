using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;
using System.IO;
using System.Text.Json;

namespace SwiftAnalyzer.Core.Services.ML
{
    public class DeBertaModel : IDisposable
    {
        private readonly InferenceSession modelSession;
        private readonly InferenceSession tokenizerSession;
        private readonly Dictionary<string, int> vocab;
        private const int MaxLength = 512;

        public DeBertaModel(string modelPath, string tokenizerPath, string vocabPath)
        {
            // Charger le vocabulaire
            vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabPath));

            // Initialiser les sessions ONNX
            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CPU();
            
            modelSession = new InferenceSession(modelPath, sessionOptions);
            tokenizerSession = new InferenceSession(tokenizerPath, sessionOptions);
        }

        public async Task<SwiftAnalysisResult> AnalyzeSwiftMessage(string text)
        {
            var result = new SwiftAnalysisResult { OriginalText = text };

            // 1. Détecter les types d'opérations principaux
            var mainTypes = await ClassifyZeroShot(text, SwiftClassificationExamples.GetMainOperationTypes());
            result.MainOperationType = mainTypes.First().Item1;

            // 2. Détecter les conditions et dépendances
            var conditions = await ClassifyZeroShot(text, SwiftClassificationExamples.GetConditionTypes());
            var significantConditions = conditions.Where(c => c.Item2 > 0.3).ToList();
            result.HasConditions = significantConditions.Any();
            result.Conditions = significantConditions;

            // 3. Détecter les dates et délais
            var timeConstraints = await ClassifyZeroShot(text, SwiftClassificationExamples.GetTimeConstraints());
            result.TimeConstraints = timeConstraints.Where(t => t.Item2 > 0.3).ToList();

            // 4. Analyser les sous-parties du message
            var sentences = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                              .Select(s => s.Trim())
                              .Where(s => !string.IsNullOrEmpty(s))
                              .ToList();

            foreach (var sentence in sentences)
            {
                var subTasks = await ClassifyZeroShot(sentence, SwiftClassificationExamples.GetSubTaskTypes());
                var significantTasks = subTasks.Where(t => t.Item2 > 0.4).ToList();
                if (significantTasks.Any())
                {
                    result.SubTasks.Add(new SubTask 
                    { 
                        Text = sentence,
                        Type = significantTasks.First().Item1,
                        Confidence = significantTasks.First().Item2,
                        RelatedTasks = significantTasks.Skip(1).ToList()
                    });
                }
            }

            return result;
        }

        public async Task<List<(string Label, float Score)>> ClassifyZeroShot(
            string premise,
            Dictionary<string, string> labelHypotheses)
        {
            var results = new List<(string Label, float Score)>();

            foreach (var (label, hypothesis) in labelHypotheses)
            {
                // Tokenize using ONNX tokenizer
                var tokenized = Tokenize(premise, hypothesis);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", tokenized.InputIds),
                    NamedOnnxValue.CreateFromTensor("attention_mask", tokenized.AttentionMask)
                };

                // Run inference
                using var outputs = modelSession.Run(inputs);
                var logits = outputs.First(x => x.Name == "logits").AsTensor<float>();
                var probabilities = Softmax(logits.ToArray());

                // For zero-shot NLI, we use the "entailment" score (index 0)
                results.Add((label, probabilities[0]));
            }

            var sum = results.Sum(r => r.Item2);
            return results.Select(r => (r.Label, r.Item2 / sum)).OrderByDescending(r => r.Item2).ToList();
        }

        private (DenseTensor<long> InputIds, DenseTensor<long> AttentionMask) Tokenize(string premise, string hypothesis)
        {
            // Préparer les entrées pour le tokenizer ONNX
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("premise", new DenseTensor<string>(new[] { premise }, new[] { 1 })),
                NamedOnnxValue.CreateFromTensor("hypothesis", new DenseTensor<string>(new[] { hypothesis }, new[] { 1 }))
            };

            // Exécuter le tokenizer
            using var outputs = tokenizerSession.Run(inputs);

            // Récupérer les tenseurs de sortie
            var inputIds = outputs.First(x => x.Name == "input_ids").AsTensor<long>();
            var attentionMask = outputs.First(x => x.Name == "attention_mask").AsTensor<long>();

            return (inputIds, attentionMask);
        }

        private float[] Softmax(float[] logits)
        {
            var maxLogit = logits.Max();
            var exp = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
            var sumExp = exp.Sum();
            return exp.Select(x => (float)(x / sumExp)).ToArray();
        }

        public void Dispose()
        {
            modelSession?.Dispose();
            tokenizerSession?.Dispose();
        }
    }

    public class SwiftAnalysisResult
    {
        public string OriginalText { get; set; } = "";
        public string MainOperationType { get; set; } = "";
        public bool HasConditions { get; set; }
        public List<(string Label, float Score)> Conditions { get; set; } = new();
        public List<(string Label, float Score)> TimeConstraints { get; set; } = new();
        public List<SubTask> SubTasks { get; set; } = new();
    }

    public class SubTask
    {
        public string Text { get; set; } = "";
        public string Type { get; set; } = "";
        public float Confidence { get; set; }
        public List<(string Label, float Score)> RelatedTasks { get; set; } = new();
    }

    public static class SwiftClassificationExamples
    {
        public static Dictionary<string, string> GetMainOperationTypes()
        {
            return new Dictionary<string, string>
            {
                ["extend_or_pay"] = "Ce message demande soit une prorogation, soit un paiement en fonction de conditions",
                ["extend"] = "Ce message demande uniquement une prorogation",
                ["pay"] = "Ce message demande uniquement un paiement",
                ["modify"] = "Ce message demande une modification des conditions",
                ["info"] = "Ce message demande des informations"
            };
        }

        public static Dictionary<string, string> GetConditionTypes()
        {
            return new Dictionary<string, string>
            {
                ["conditional_payment"] = "Le message contient une condition de paiement dépendant d'autres facteurs",
                ["conditional_extension"] = "Le message contient une condition de prorogation dépendant d'autres facteurs",
                ["alternative_action"] = "Le message propose des actions alternatives selon certaines conditions",
                ["deadline_dependent"] = "L'action demandée dépend d'une date limite"
            };
        }

        public static Dictionary<string, string> GetTimeConstraints()
        {
            return new Dictionary<string, string>
            {
                ["specific_date"] = "Le message mentionne une date spécifique pour l'action",
                ["relative_date"] = "Le message mentionne un délai relatif (par exemple, dans X jours)",
                ["deadline"] = "Le message mentionne une date limite",
                ["conditional_date"] = "La date mentionnée dépend d'une condition"
            };
        }

        public static Dictionary<string, string> GetSubTaskTypes()
        {
            return new Dictionary<string, string>
            {
                ["extension_request"] = "Cette partie demande une prorogation",
                ["payment_instruction"] = "Cette partie donne des instructions de paiement",
                ["condition_statement"] = "Cette partie définit une condition",
                ["deadline_specification"] = "Cette partie spécifie un délai ou une date limite",
                ["fallback_action"] = "Cette partie décrit l'action à prendre en cas de non-respect des conditions principales"
            };
        }
    }
}
