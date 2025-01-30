# SWIFTY

SWIFTY est un analyseur intelligent de messages SWIFT 799 utilisant l'IA pour détecter et classifier les tâches et conditions dans les messages.

## Fonctionnalités

- Détection des types d'opérations (prorogation, paiement, etc.)
- Identification des conditions et dépendances
- Analyse des contraintes temporelles
- Support multilingue (FR/EN) via mDeBERTa
- Extraction intelligente de la partie narrative des messages

## Installation

1. Cloner le repository
```bash
git clone https://github.com/PocKeTio/swifty.git
cd swifty
```

2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation

```python
import asyncio
from swift_analyzer import Swift799Analyzer

async def analyze_message():
    # Initialiser l'analyseur
    analyzer = Swift799Analyzer(extract_narrative_only=True)
    
    # Message SWIFT à analyser
    message = """
    {1:F01BANKBEBBAXXX0057000289}{2:I799BANKITRXXXXXN}{4:
    20:123456
    79:NOTRE GARANTIE NO 123
    NOUS VOUS PRIONS DE BIEN VOULOIR PROROGER LA VALIDITE
    DE LA GARANTIE JUSQU'AU 31.12.2025 OU PROCEDER AU
    PAIEMENT SI LE BENEFICIAIRE LE DEMANDE.
    -}
    """
    
    # Analyser le message
    result = await analyzer.analyze_message(message)
    
    # Afficher les résultats
    print(f"Type principal : {result.main_task}")
    print(f"Actions conditionnelles : {result.has_conditional_actions}")
    
    for task in result.sub_tasks:
        print(f"Sous-tâche : {task.type} ({task.confidence:.2%})")
        
    for constraint in result.time_constraints:
        print(f"Contrainte temporelle : {constraint.type} ({constraint.confidence:.2%})")

# Exécuter l'analyse
asyncio.run(analyze_message())
```

## Architecture

Le projet utilise :
- mDeBERTa-v3-base-mnli-xnli pour l'analyse zero-shot
- PyTorch pour l'inférence
- asyncio pour les opérations asynchrones

## Structure du Projet

```
swifty/
├── requirements.txt
├── README.md
└── swift_analyzer/
    ├── __init__.py
    ├── models.py              # Classes de données
    ├── swift799_analyzer.py   # Analyseur principal
    └── ml/
        └── deberta_model.py   # Modèle ML et inférence
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.
