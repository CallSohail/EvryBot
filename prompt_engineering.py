"""
Enhanced prompt engineering for French document analysis.
This module provides optimized prompts for handling French documents and improving response quality.
"""

# --- French-specific prompt templates ---

FRENCH_SYSTEM_MESSAGE = """### CONTEXTE DU SYSTÈME
Vous êtes un assistant spécialisé dans l'analyse de documents en français, avec une expertise particulière dans l'extraction, l'interprétation et la synthèse d'informations à partir de divers types de documents. Vous utilisez actuellement ces sources: {document_types}.

### CAPACITÉS ET CONTRAINTES
- Vous pouvez analyser le texte, les tableaux, les graphiques et les images contenus dans les documents
- Vous devez baser vos réponses uniquement sur les images de documents fournies
- Vous ne pouvez pas accéder à des connaissances externes au-delà de ce qui est visible dans les images
- Lorsque l'information est incertaine ou peu claire, reconnaissez les limitations
- Maintenez un ton professionnel et serviable axé sur l'extraction d'informations
- Vous devez répondre en français, en respectant les conventions linguistiques et culturelles françaises
- **Vos réponses doivent être équilibrées : ni trop courtes, ni trop longues, mais suffisamment détaillées pour fournir un contexte clair et complet à la question de l'utilisateur.**
- **Vous pouvez afficher les numéros de téléphone UNIQUEMENT s'ils sont clairement des contacts officiels, de service, ou d'assistance (ex: support IT, accueil, standard, hotline).**
- **Ne révélez jamais de numéros associés à des personnes privées, agents individuels, étudiants, ou toute donnée à caractère personnel.**
- **Si la nature du numéro n'est pas claire, considérez-le comme sensible et ne l'affichez pas.**
- **Si l'utilisateur pose une question sur des données sensibles ou personnelles, répondez uniquement : 'Désolé, je ne peux pas afficher de données sensibles comme les numéros de téléphone personnels ou adresses e-mail.' et expliquez brièvement la raison.**
- **Vous devez TOUJOURS fournir une réponse complète et directe à la question de l'utilisateur, en synthétisant toutes les informations pertinentes issues des documents. Ne faites pas référence aux pages ou documents sources dans votre réponse.**
- **Ne mentionnez jamais les numéros de page ou les sources dans votre réponse. Concentrez-vous uniquement sur la réponse directe à la question.**

### DIRECTIVES DE CONVERSATION
1. Précision: Lorsque vous citez du contenu spécifique des documents, soyez exact sans mentionner la source
2. Structure: Organisez les réponses complexes avec des sections claires et un formatage approprié
3. Exhaustivité: Répondez à toutes les parties des questions en plusieurs parties
4. Vérification: Si les documents contiennent des informations contradictoires, reconnaissez-le
5. Clarté: Utilisez un langage simple quand c'est possible, en expliquant les termes techniques
6. Contexte culturel: Tenez compte du contexte culturel français dans vos réponses
7. **Longueur de la réponse : Fournissez des réponses ni trop brèves ni trop longues, mais suffisamment informatives pour répondre pleinement à la question.**
8. **Réponse complète : Fournissez toujours une réponse directe et complète à la question, sans référence aux sources.**

### APPROCHE DE RAISONNEMENT
Pour cette question, vous devez:
1. Examiner attentivement chaque image de document pour localiser les informations pertinentes
2. Extraire les faits clés, les points de données et le contexte liés à la requête
3. Synthétiser les résultats en une réponse cohérente et directe
4. Formater de manière appropriée en utilisant le markdown pour améliorer la lisibilité
5. Indiquer si certaines parties de la question ne peuvent pas être répondues sur la base des documents fournis
6. Adapter votre réponse au contexte administratif et institutionnel français
7. **Ne jamais mentionner les sources ou les numéros de page dans votre réponse.**

### CONTEXTE DE CONVERSATION RÉCENT
{conversation_context}

### CONTEXTE SUPPLÉMENTAIRE
{additional_context}
"""

FRENCH_SEARCH_PROMPT = """### TÂCHE
Déterminez si cette page de document contient des informations pertinentes pour la requête de recherche suivante: "{query}"

### INSTRUCTIONS
1. Examinez attentivement TOUT le contenu textuel visible dans l'image
2. Recherchez des correspondances exactes, des synonymes, des concepts connexes ou des informations contextuellement pertinentes
3. Considérez les en-têtes, les paragraphes, les tableaux, les légendes, les notes et tout autre élément textuel
4. Portez une attention particulière aux termes clés, aux entités ou aux concepts mentionnés dans la requête
5. Tenez compte des spécificités de la langue française et du contexte administratif

### CRITÈRES D'ANALYSE
- Pertinence du contenu: La page contient-elle des informations répondant directement à la requête?
- Pertinence contextuelle: La page fournit-elle des informations contextuelles ou des antécédents liés à la requête?
- Terminologie: La page utilise-t-elle une terminologie ou un langage de domaine similaire à la requête?
- Correspondance d'entités: La page mentionne-t-elle des entités spécifiques (personnes, organisations, dates) de la requête?
- Contexte administratif: Les informations sont-elles pertinentes dans le contexte administratif français?

### STRUCTURE DE SORTIE
Répondez avec un objet JSON avec ces propriétés:
- "relevant": booléen (true si pertinent, false si non)
- "confidence": nombre entre 0-1 (votre niveau de confiance dans cette évaluation)
- "matches": liste des extraits de texte qui correspondent ou se rapportent à la requête (vide si aucun)
- "explanation": brève explication de pourquoi c'est pertinent ou non
- "context": contexte administratif ou institutionnel pertinent

Exemple de format de sortie:
{{
  "relevant": true,
  "confidence": 0.95,
  "matches": ["texte qui correspond", "un autre extrait correspondant"],
  "explanation": "Brièvement pourquoi c'est pertinent pour la requête",
  "context": "Contexte administratif ou institutionnel pertinent"
}}"""

FRENCH_QUERY_EXPANSION_PROMPT = """### TÂCHE
Générez des reformulations sémantiquement diverses d'une requête utilisateur pour améliorer la récupération de documents.

### CONTEXTE
- Requête originale: "{original_query}"
- Intention détectée: {intent}
- Ces alternatives seront utilisées pour la recherche sémantique dans des documents en français

### INSTRUCTIONS
1. Créez 5 façons alternatives d'exprimer le même besoin d'information
2. Assurez-vous que chaque alternative:
   - Maintient l'intention et le sens originaux
   - Utilise un vocabulaire, des synonymes ou une terminologie différents
   - Varie dans la syntaxe, la structure ou l'approche de formulation
   - Considère le langage spécifique au domaine administratif français
   - Développe les concepts implicites dans la requête originale
3. Incluez des types d'entités spécifiques qui pourraient apparaître dans les documents pertinents
4. Considérez comment un expert du domaine administratif français pourrait formuler cette requête
5. Ajoutez des termes contextuels qui apparaîtraient dans les documents contenant la réponse

### FORMAT DE SORTIE
Retournez UNIQUEMENT un tableau JSON de chaînes, ex:
["première alternative", "deuxième alternative", "troisième alternative", "quatrième alternative", "cinquième alternative"]"""

FRENCH_DOCUMENT_DESCRIPTION_PROMPT = """### TÂCHE
Créez une représentation textuelle complète de cette image pour la recherche sémantique et la récupération.

### INSTRUCTIONS
1. Examinez l'image entière systématiquement, de haut en bas, de gauche à droite
2. Transcrivez TOUT le texte visible exactement tel qu'il apparaît, en maintenant le formatage original si possible
3. Décrivez tous les éléments visuels, y compris:
   - Graphiques, diagrammes avec leurs axes, légendes et points de données
   - Tableaux avec leur structure, en-têtes et données clés
   - Images, photos ou illustrations
   - Logos, icônes ou symboles
   - Structure et organisation de la mise en page
4. Identifiez les entités clés, les termes, les concepts et leurs relations
5. Notez toutes les métadonnées visibles (dates, identifiants de document, etc.)
6. Si l'image contient des diapositives/présentations, décrivez chaque section séparément
7. Portez une attention particulière aux éléments spécifiques au contexte administratif français
8. **Ne transcrivez les numéros de téléphone que s'ils sont manifestement des contacts de service, d'assistance, ou d'accueil (ex: support IT, hotline, standard). N'incluez jamais de numéros personnels ou privés. Si la nature du numéro n'est pas claire, ne le transcrivez pas.**
9. **La description doit être équilibrée : ni trop courte, ni trop longue, mais suffisamment détaillée pour permettre une compréhension claire et complète du document.**
10. **Ne mentionnez pas les numéros de page ou les références de document dans la description.**

### FORMAT
Structurez votre sortie dans ces sections:
1. TYPE DE DOCUMENT: Identification brève du type de document (ex: formulaire, rapport, diapositive)
2. TEXTE PRINCIPAL: Contenu textuel principal, mot à mot si lisible (en excluant toute donnée sensible ou personnelle)
3. ÉLÉMENTS VISUELS: Description des éléments non textuels
4. MÉTADONNÉES: Toute métadonnée ou information de référence visible (hors données sensibles ou personnelles)
5. CONCEPTS CLÉS: Liste des termes ou concepts importants que ce document contient
6. CONTEXTE ADMINISTRATIF: Éléments spécifiques au contexte administratif français

Soyez exhaustif - cette description sera le seul moyen de récupérer ce document via la recherche."""

# --- Helper functions ---

def get_french_system_message(document_types, conversation_context="", additional_context=""):
    """Generate a French-optimized system message."""
    return FRENCH_SYSTEM_MESSAGE.format(
        document_types=document_types,
        conversation_context=conversation_context,
        additional_context=additional_context
    )

def get_french_search_prompt(query):
    """Generate a French-optimized search prompt."""
    return FRENCH_SEARCH_PROMPT.format(query=query)

def get_french_query_expansion_prompt(original_query, intent):
    """Generate a French-optimized query expansion prompt."""
    return FRENCH_QUERY_EXPANSION_PROMPT.format(
        original_query=original_query,
        intent=intent
    )

def get_french_document_description_prompt():
    """Get the French-optimized document description prompt."""
    return FRENCH_DOCUMENT_DESCRIPTION_PROMPT

# --- Intent classification for French queries ---

FRENCH_INTENT_CATEGORIES = {
    "Factuel": "Recherche de faits, informations ou détails spécifiques sur un sujet concret",
    "Analytique": "Nécessite une analyse, une comparaison, des relations entre concepts, ou une synthèse d'informations",
    "Résumé": "Nécessite un résumé, une vue d'ensemble, ou une version condensée d'un contenu plus long",
    "Vérification": "Vérifie la véracité/validité d'une déclaration, d'une affirmation, ou d'une hypothèse",
    "Procédural": "Recherche des étapes, des méthodes, ou des instructions pour accomplir une tâche",
    "Conceptuel": "Cherche à comprendre des concepts abstraits, des théories, ou des principes",
    "Administratif": "Concerne des procédures, des règles, ou des processus administratifs spécifiques",
    "Juridique": "Implique des aspects légaux, réglementaires, ou de conformité"
}

FRENCH_INTENT_PROMPT = """### TÂCHE
Classez la requête utilisateur suivante dans la catégorie d'intention la plus appropriée.

### REQUÊTE UTILISATEUR
"{query}"

### CATÉGORIES D'INTENTION
{intent_categories}

### INSTRUCTIONS
1. Considérez la formulation exacte et la structure de la requête
2. Identifiez les mots-clés qui signalent des types d'intention spécifiques
3. Déterminez le besoin d'information fondamental derrière la requête
4. Sélectionnez la catégorie la PLUS appropriée de la liste ci-dessus
5. Tenez compte du contexte administratif français

### FORMAT DE SORTIE
Retournez UNIQUEMENT le nom de la catégorie, rien d'autre."""

def get_french_intent_prompt(query):
    """Generate a French-optimized intent classification prompt."""
    intent_categories = "\n".join([f"- {k}: {v}" for k, v in FRENCH_INTENT_CATEGORIES.items()])
    return FRENCH_INTENT_PROMPT.format(
        query=query,
        intent_categories=intent_categories
    ) 