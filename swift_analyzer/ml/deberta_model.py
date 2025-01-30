from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..models import Swift799Message, SubTask, TimeConstraint

class DeBertaModel:
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    async def analyze_swift_message(self, text: str) -> Swift799Message:
        result = Swift799Message(
            original_text=text,
            main_task="",
            has_conditional_actions=False
        )

        # 1. Détecter les types d'opérations principaux
        main_types = await self.classify_zero_shot(text, self.get_main_operation_types())
        result.main_task = main_types[0][0]

        # 2. Détecter les conditions et dépendances
        conditions = await self.classify_zero_shot(text, self.get_condition_types())
        significant_conditions = [(label, score) for label, score in conditions if score > 0.3]
        result.has_conditional_actions = len(significant_conditions) > 0

        # 3. Détecter les dates et délais
        time_constraints = await self.classify_zero_shot(text, self.get_time_constraints())
        for label, score in time_constraints:
            if score > 0.3:
                result.time_constraints.append(TimeConstraint(type=label, confidence=score))

        # 4. Analyser les sous-parties du message
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for sentence in sentences:
            sub_tasks = await self.classify_zero_shot(sentence, self.get_sub_task_types())
            significant_tasks = [(label, score) for label, score in sub_tasks if score > 0.4]
            
            if significant_tasks:
                main_label, main_score = significant_tasks[0]
                result.sub_tasks.append(SubTask(
                    text=sentence,
                    type=main_label,
                    confidence=main_score,
                    related_tasks=significant_tasks[1:]
                ))

        return result

    async def classify_zero_shot(self, premise: str, label_hypotheses: Dict[str, str]) -> List[Tuple[str, float]]:
        results = []
        
        for label, hypothesis in label_hypotheses.items():
            # Tokenize
            inputs = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(**inputs)
                probs = torch.softmax(output.logits[0], -1)
                # Pour le zero-shot NLI, nous utilisons le score "entailment" (index 0)
                score = probs[0].item()
                results.append((label, score))

        # Normaliser les scores
        total_score = sum(score for _, score in results)
        normalized_results = [(label, score/total_score) for label, score in results]
        
        # Trier par score décroissant
        return sorted(normalized_results, key=lambda x: x[1], reverse=True)

    @staticmethod
    def get_main_operation_types() -> Dict[str, str]:
        return {
            "extend_or_pay": "Ce message demande soit une prorogation, soit un paiement en fonction de conditions",
            "extend": "Ce message demande uniquement une prorogation",
            "pay": "Ce message demande uniquement un paiement",
            "modify": "Ce message demande une modification des conditions",
            "info": "Ce message demande des informations"
        }

    @staticmethod
    def get_condition_types() -> Dict[str, str]:
        return {
            "conditional_payment": "Le message contient une condition de paiement dépendant d'autres facteurs",
            "conditional_extension": "Le message contient une condition de prorogation dépendant d'autres facteurs",
            "alternative_action": "Le message propose des actions alternatives selon certaines conditions",
            "deadline_dependent": "L'action demandée dépend d'une date limite"
        }

    @staticmethod
    def get_time_constraints() -> Dict[str, str]:
        return {
            "specific_date": "Le message mentionne une date spécifique pour l'action",
            "relative_date": "Le message mentionne un délai relatif (par exemple, dans X jours)",
            "deadline": "Le message mentionne une date limite",
            "conditional_date": "La date mentionnée dépend d'une condition"
        }

    @staticmethod
    def get_sub_task_types() -> Dict[str, str]:
        return {
            "extension_request": "Cette partie demande une prorogation",
            "payment_instruction": "Cette partie donne des instructions de paiement",
            "condition_statement": "Cette partie définit une condition",
            "deadline_specification": "Cette partie spécifie un délai ou une date limite",
            "fallback_action": "Cette partie décrit l'action à prendre en cas de non-respect des conditions principales"
        }
