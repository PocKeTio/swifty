import re
from typing import Optional
from .ml.deberta_model import DeBertaModel
from .models import Swift799Message

class Swift799Analyzer:
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", extract_narrative_only: bool = False):
        self.model = DeBertaModel(model_name)
        self.extract_narrative_only = extract_narrative_only

    async def analyze_message(self, message_text: str) -> Swift799Message:
        # Prétraitement optionnel pour extraire la partie narrative
        text_to_analyze = self.extract_narrative_part(message_text) if self.extract_narrative_only else message_text
        
        # Analyser le message avec le modèle
        return await self.model.analyze_swift_message(text_to_analyze)

    def extract_narrative_part(self, message_text: str) -> str:
        """Extrait la partie narrative d'un message SWIFT."""
        
        # Recherche de la partie narrative après les en-têtes et codes
        narrative_match = re.search(r'(?:^|\n)\s*77E:?\s*(.+?)(?:\n|$)', message_text, re.DOTALL)
        if narrative_match:
            return narrative_match.group(1).strip()

        # Si pas de champ 77E, chercher après le dernier champ numéroté
        last_field_match = re.search(r'(?:^|\n)\s*\d{2}[A-Z]:?\s*(.+?)(?:\n|$)', message_text, re.DOTALL | re.RIGHTMOST)
        if last_field_match:
            return last_field_match.group(1).strip()

        # Si aucun format reconnu, retourner le texte complet
        return message_text.strip()
