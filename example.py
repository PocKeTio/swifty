import asyncio
from swift_analyzer import Swift799Analyzer

async def main():
    # Initialiser l'analyseur
    analyzer = Swift799Analyzer(extract_narrative_only=True)
    
    # Message SWIFT à analyser
    message = """{1:F01BANKBEBBAXXX0057000289}{2:I799BANKITRXXXXXN}{4:
    20:123456
    79:NOTRE GARANTIE NO 123
    NOUS VOUS PRIONS DE BIEN VOULOIR PROROGER LA VALIDITE
    DE LA GARANTIE JUSQU'AU 31.12.2025 OU PROCEDER AU
    PAIEMENT SI LE BENEFICIAIRE LE DEMANDE.
    -}"""
    
    try:
        # Analyser le message
        result = await analyzer.analyze_message(message)
        
        # Afficher les résultats
        print("\nRésultat de l'analyse :")
        print("-" * 50)
        print(f"Type principal : {result.main_task}")
        print(f"Actions conditionnelles : {result.has_conditional_actions}")
        
        if result.sub_tasks:
            print("\nSous-tâches détectées :")
            for task in result.sub_tasks:
                print(f"- {task.type} ({task.confidence:.2%})")
                print(f"  Texte : {task.text}")
                if task.related_tasks:
                    print("  Tâches liées :")
                    for related_type, score in task.related_tasks:
                        print(f"    - {related_type} ({score:.2%})")
        
        if result.time_constraints:
            print("\nContraintes temporelles :")
            for constraint in result.time_constraints:
                print(f"- {constraint.type} ({constraint.confidence:.2%})")
    
    except Exception as e:
        print(f"Erreur lors de l'analyse : {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
