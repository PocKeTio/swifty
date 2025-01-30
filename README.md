# SWIFTY

SWIFTY est un analyseur intelligent de messages SWIFT 799 utilisant l'IA pour détecter et classifier les tâches et conditions dans les messages.

## Fonctionnalités

- Détection des types d'opérations (prorogation, paiement, etc.)
- Identification des conditions et dépendances
- Analyse des contraintes temporelles
- Support multilingue (FR/EN) via mDeBERTa
- Extraction intelligente de la partie narrative des messages

## Architecture

Le projet utilise :
- mDeBERTa-v3-base-mnli-xnli pour l'analyse zero-shot
- ONNX Runtime pour l'inférence optimisée
- .NET Core pour l'implémentation

## Installation

1. Cloner le repository
2. Télécharger le modèle mDeBERTa converti en ONNX
3. Référencer le projet dans votre solution

## Utilisation

```csharp
// Initialiser l'analyseur
using var analyzer = new Swift799Analyzer("path/to/model.onnx", extractNarrativeOnly: true);

// Analyser un message
var result = await analyzer.AnalyzeMessage("Votre message SWIFT ici");

// Accéder aux résultats
Console.WriteLine($"Type principal : {result.MainTask}");
Console.WriteLine($"Actions conditionnelles : {result.HasConditionalActions}");

foreach (var task in result.SubTasks)
{
    Console.WriteLine($"Sous-tâche : {task.Type} ({task.Confidence:P2})");
}
```

## Préparation du modèle

Pour convertir le modèle mDeBERTa en ONNX :

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_graph_to_onnx import convert

model_id = "microsoft/mdeberta-v3-base-mnli-xnli"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Conversion en ONNX
convert(framework="pt", 
        model=model, 
        tokenizer=tokenizer, 
        output="mdeberta.onnx",
        opset=12)

# Optionnel : Quantification pour réduire la taille
# Utiliser onnxruntime.quantization
```
