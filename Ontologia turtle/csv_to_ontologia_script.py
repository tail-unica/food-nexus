"""
File con varie funzioni e dati necessari per la conversione dei dati di 
HUMMUS e open food facts in un formato RDF
"""


from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD, URIRef #type: ignore
import pandas as pd #type: ignore
import re
import ollama #type: ignore
import csv


def sanitize_for_uri(value) -> str:
    """
    Funzione di sanificazione generica per URI 
    (alcuni nomi di hummus iniziavano con simboli speciali)

    :param value: valore da sanitizzare

    :return: valore sanitizzato
    """
    return re.sub(r"[^a-zA-Z0-9]", "", str(value)) 


def clean_column_name(col_name) -> str:
    """
    Funzione per pulire il nome della colonna, rimuovendo contenuti tra parentesi quadre

    :param col_name: nome della colonna

    :return: nome della colonna pulito
    """
    return re.sub(r'\s*\[.*?\]', '', col_name).strip()


def crea_namespace() -> None:
    """
    Funzione per creare il file ttl con il namespace personalizzato di unica
    """

    # Creiamo il grafo
    g = Graph()

    # Definizione dei namespace
    SCHEMA = Namespace("https://schema.org/")
    UNICA = Namespace("file:///../File_CSV/namespace_unica.ttl#")  # Namespace personalizzato, poi si decidera il nome e l'indirizzo
    XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")
    RDFS_NS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

    # Aggiungiamo i prefissi al grafo
    g.bind("schema", SCHEMA)
    g.bind("unica", UNICA) 
    g.bind("xsd", XSD_NS)
    g.bind("rdfs", RDFS_NS)
    g.bind("rdf", RDF_NS)

    # Creazione delle classi
    g.add((UNICA.FoodProducer, RDF.type, RDFS.Class))
    g.add((UNICA.FoodProducer, RDFS.subClassOf, SCHEMA.Organization))
    g.add((UNICA.FoodProducer, RDFS.label, Literal("Food Producer", lang="en")))
    g.add((UNICA.FoodProducer, RDFS.comment, Literal("Information about the producer of a food item.", lang="en")))

    g.add((UNICA.UserConstraint, RDF.type, RDFS.Class))
    g.add((UNICA.UserConstraint, RDFS.subClassOf, SCHEMA.Intangible))
    g.add((UNICA.UserConstraint, RDFS.label, Literal("User Constraint", lang="en")))
    g.add((UNICA.UserConstraint, RDFS.comment, Literal("Constraint about what a user can or want to eat.", lang="en")))

    # Proprietà per UserConstraint
    g.add((UNICA.constraintName, RDF.type, RDF.Property))
    g.add((UNICA.constraintName, RDFS.domain, UNICA.UserConstraint))
    g.add((UNICA.constraintName, RDFS.range, XSD.string))
    g.add((UNICA.constraintName, RDFS.label, Literal("Constraint Name", lang="en")))
    g.add((UNICA.constraintName, RDFS.comment, Literal("Name of the constraint.", lang="en")))

    g.add((UNICA.constraintDescription, RDF.type, RDF.Property))
    g.add((UNICA.constraintDescription, RDFS.domain, UNICA.UserConstraint))
    g.add((UNICA.constraintDescription, RDFS.range, XSD.string))
    g.add((UNICA.constraintDescription, RDFS.label, Literal("Constraint Description", lang="en")))
    g.add((UNICA.constraintDescription, RDFS.comment, Literal("Description of the constraint.", lang="en")))

    # Creazione della classe Indicator
    g.add((UNICA.Indicator, RDF.type, RDFS.Class))
    g.add((UNICA.Indicator, RDFS.subClassOf, SCHEMA.NutritionInformation))
    g.add((UNICA.Indicator, RDFS.label, Literal("Indicator", lang="en")))
    g.add((UNICA.Indicator, RDFS.comment, Literal("Represents nutritional and sustainability indicators for food items.", lang="en")))

    # Proprietà per Indicator
    proprieta_indicator = [
        ("calciumPer100g", "Calcium per 100g (mg)", "Calcium content per 100 grams.", XSD.float),
        ("ironPer100g", "Iron per 100g (mg)", "Iron content per 100 grams.", XSD.float),
        ("vitaminCPer100g", "Vitamin C per 100g (mg)", "Vitamin C content per 100 grams.", XSD.float),
        ("vitaminAPer100g", "Vitamin A per 100g (mcg)", "Vitamin A content per 100 grams.", XSD.float),
        ("whoScore", "WHO Score", "A score indicating WHO healthfulness assessment.", XSD.float),
        ("fsaScore", "FSA Score", "A score based on the Food Standards Agency's healthfulness criteria.", XSD.float),
        ("nutriScore", "Nutri-Score", "A score indicating the nutritional value based on the Nutri-Score system.", XSD.float),
        ("nutriscoreScore", "Nutri-Score Score", "A numeric score for Nutri-Score system.", XSD.float),
        ("ecoscoreScore", "Eco-Score Score", "A numeric score for Eco-Score system.", XSD.float),
        ("nutritionScoreFrPer100g", "Nutrition Score FR per 100g", "French nutrition score per 100 grams.", XSD.float),
        ("nutriscoreGrade", "Nutri-Score Grade", "A grade indicating the Nutri-Score nutritional level.", XSD.string),
        ("ecoscoreGrade", "Eco-Score Grade", "A grade indicating the Eco-Score environmental impact level.", XSD.string),
        ("novaGroup", "NOVA Group", "Group classification of processing level based on NOVA system.", XSD.string)
    ]

    for prop, label, comment, range_type in proprieta_indicator:
        g.add((UNICA[prop], RDF.type, RDF.Property))
        g.add((UNICA[prop], RDFS.domain, UNICA.Indicator))
        g.add((UNICA[prop], RDFS.range, range_type))
        g.add((UNICA[prop], RDFS.label, Literal(label, lang="en")))
        g.add((UNICA[prop], RDFS.comment, Literal(comment, lang="en")))

    # Salva in formato Turtle con il nome 'namespace_unica.ttl'
    with open("../File_CSV/namespace_unica.ttl", "w") as f:
        f.write(g.serialize(format="turtle"))
    
    print(f"namespace creato con successo")


def converti_hummus_in_rdf() -> None:
    """
    Funzione per convertire i dati di hummus in un formato RDF
    """

    # Inizializza il grafo e i namespace
    g = Graph()

    # Ottiene namespace 
    EX = Namespace("../File_CSV/namespace_unica.ttl/")
    SCHEMA = Namespace("https://schema.org/")

    # Associa i namespace al grafo
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)

    # File di input
    #file_ricette = "../File_CSV/pp_recipes.csv" 
    #file_review = "../File_CSV/pp_reviews.csv"
    #file_utenti = "../File_CSV/pp_members.csv"
    file_ricette = "../File_CSV/pp_recipes_righe.csv"
    file_review = "../File_CSV/pp_reviews_righe.csv"
    file_utenti = "../File_CSV/pp_members_righe.csv"
    file_output = "../File_CSV/ontologia_nostra_hummus.ttl"

    # Carica i CSV
    df_ricette = pd.read_csv(file_ricette)
    df_review = pd.read_csv(file_review)
    df_utenti = pd.read_csv(file_utenti)

    tag_count = {}
    ingredient_count = {}

    # Indicatori nutrizionali
    indicator_fields = {
        'servingSize': 'grams', 'calories': 'cal', 'caloriesFromFat': 'cal',
        'totalFat': 'grams', 'saturatedFat': 'grams', 'cholesterol': 'mg', 'sodium': 'mg',
        'totalCarbohydrate': 'grams', 'dietaryFiber': 'grams', 'sugars': 'grams', 'protein': 'grams',
        'who_score': None, 'fsa_score': None, 'nutri_score': None
    }



    # Crea le entità UserGroup
    for idx, row in df_utenti.iterrows():
        if pd.notna(row['member_id']):
            group_id = URIRef(EX[f"UserGroup_{sanitize_for_uri(row['member_id'])}"])
            g.add((group_id, RDF.type, SCHEMA.Organization))
            g.add((group_id, SCHEMA.name, Literal(row['member_name'], lang="en")))

    # Crea le entità Recipe
    for idx, row in df_ricette.iterrows():
        if pd.notna(row['recipe_id']):

            recipe_id = URIRef(EX[f"Recipe_{sanitize_for_uri(row['recipe_id'])}"])
            g.add((recipe_id, RDF.type, SCHEMA.Recipe))
            g.add((recipe_id, SCHEMA.name, Literal(row['title'], lang="en")))
            g.add((recipe_id, SCHEMA.identifier, Literal(row['recipe_id'])))
            g.add((recipe_id, SCHEMA.RecipeInstructions, Literal(row['directions'], lang="en")))

            author_id = URIRef(EX[f"UserGroup_{sanitize_for_uri(row['author_id'])}"]) 
            g.add((author_id, SCHEMA.publishesRecipe, recipe_id))

            # UserConstraint
            if pd.notna(row['tags']):
                tags = str(row['tags']).strip("[]").replace("'", "").split(sep=', ')
                for tag in tags:
                    tag = sanitize_for_uri(tag)
                    tag_id = URIRef(EX[f"Tag_{sanitize_for_uri(tag)}"])
                    if tag not in tag_count:
                        tag_count[tag] = 1
                        g.add((tag_id, RDF.type, EX.UserConstraint))
                        g.add((tag_id, SCHEMA.constraintName, Literal(tag, lang="en")))
                        g.add((tag_id, SCHEMA.constraintDescription, Literal(f"is a constraint about {tag}", lang="en")))

                    g.add((tag_id, SCHEMA.suitableForDiet, recipe_id))


            # Indicator
            for col, unit in indicator_fields.items():
                # Trova la colonna nel CSV corrispondente all'indicatore, dopo aver rimosso le parentesi quadre
                csv_column = next(
                    (csv_col for csv_col in row.index if clean_column_name(csv_col) == col),
                    None
                )
                # Continua solo se abbiamo trovato una corrispondenza e il valore non è NaN
                if csv_column and pd.notna(row[csv_column]):
                    indicator_id = URIRef(EX[f"Indicator_{sanitize_for_uri(col)}_{sanitize_for_uri(row['recipe_id'])}"])
                    g.add((indicator_id, RDF.type, EX.Indicator))
                    g.add((indicator_id, SCHEMA.type, Literal(col)))
                    if unit:
                        g.add((indicator_id, SCHEMA.unitText, Literal(unit)))
                    g.add((indicator_id, SCHEMA.quantity, Literal(row[csv_column], datatype=XSD.float)))
                    g.add((recipe_id, SCHEMA.NutritionInformation, indicator_id))

                
            
            # Ingredienti
            if pd.notna(row['ingredient_food_kg_names']):
                ingredients = row['ingredient_food_kg_names'].split(', ')
                for ing in ingredients:
                    ingredient_id = URIRef(EX[f"RecipeIngredient_{sanitize_for_uri(ing.replace(' ', '_').lower())}"])
                    if ing not in ingredient_count:
                        ingredient_count[ing] = 1
                        g.add((ingredient_id, RDF.type, SCHEMA.Recipe))
                        g.add((ingredient_id, SCHEMA.identifier, Literal(ingredient_id)))
                        g.add((ingredient_id, SCHEMA.RecipeInstructions, Literal(None, lang="en")))

                    ingredient_id_for_recipe = ingredient_id + str(idx)
                    g.add((ingredient_id_for_recipe, RDF.type, SCHEMA.QuantitativeValue))
                    g.add((ingredient_id_for_recipe, SCHEMA.value, Literal(None, lang="en")))
                    g.add((ingredient_id_for_recipe, SCHEMA.unitText, Literal(None, lang="en")))

                    g.add((ingredient_id, SCHEMA.hasPart, ingredient_id_for_recipe))
                    g.add((ingredient_id_for_recipe, SCHEMA.isPartOf, recipe_id))


    # Crea le entità UserReview e relazioni
    for idx, row in df_review.iterrows():
        if pd.notna(row['rating']) and isinstance(row['rating'], (int, float)):
            review_id = URIRef(EX[f"Review_{sanitize_for_uri(idx)}"])
            rating = row['rating'] / 10  # Scala ad 1-5 perche su hummus sono da 1 a 10, ma su schema da 1 a 5
            g.add((review_id, RDF.type, SCHEMA.UserReview))
            g.add((review_id, SCHEMA.reviewBody, Literal(row['text'], lang="en")))
            g.add((review_id, SCHEMA.reviewRating, Literal(rating, datatype=XSD.float)))

            g.add((review_id, SCHEMA.itemReviewed, URIRef(EX[f"Recipe_{sanitize_for_uri(row['recipe_id'])}"])))
            g.add((URIRef(EX[f"UserGroup_{sanitize_for_uri(row['member_id'])}"]), SCHEMA.publishesReview, review_id))

    # Salva il grafo RDF in formato Turtle
    g.serialize(destination=file_output, format="turtle")
    print(f"File generato con successo: {file_output}")


def converti_off_in_rdf() -> None:
    """
    Funzione per convertire i dati di off in un formato RDF
    """

    # Inizializza il grafo e i namespace
    g = Graph()

    # Definisci i namespace
    EX = Namespace("../File_CSV/namespace_unica.ttl/")
    SCHEMA = Namespace("https://schema.org/")

    # Associa i namespace al grafo
    EX = Namespace("../File_CSV/namespace_unica.ttl/")
    g.bind("schema", SCHEMA)

    # File di input e output
    file_input1 = '../File_CSV/en.openfoodfacts.org.products.csv'
    file_input = '../File_CSV/off_righe.csv'
    file_output = '../File_CSV/ontologia_nostra_off.ttl'

    # Carica il CSV
    df = pd.read_csv(file_input, sep='\t')  # Usa il separatore corretto

    qualitatives_indicators = ['nutriscore_grade', 'nova_group', 'ecoscore_grade', 'ecoscore_score']

    # Itera attraverso ogni riga per creare le entità Recipe e Indicator
    for idx, row in df.iterrows():
        if pd.notna(row['product_name']) and row['product_name'].strip() != "":

            # Crea la ricetta
            recipe_id = URIRef(EX[f"Recipe_off_{idx}"]) #ho messo off per differenziarle dagli id di hummus
            g.add((recipe_id, RDF.type, SCHEMA.Recipe))
            g.add((recipe_id, SCHEMA.name, Literal(row['product_name'], lang="en")))
            g.add((recipe_id, SCHEMA.identifier, Literal(idx, datatype=XSD.integer)))

            # Indicator
            for column in df.columns:
                if "100g" in column or column in qualitatives_indicators:
                    indicator_value = row[column]

                    if pd.notna(indicator_value) and indicator_value != "unknown":
                        
                        # Crea l'indicatore
                        indicator_id = URIRef(EX[f"{sanitize_for_uri(re.sub('_100g', '', column))}_{idx}"])
                        g.add((indicator_id, RDF.type, EX.Indicator))
                        g.add((indicator_id, SCHEMA.type, Literal(column)))
                        
                        if column not in qualitatives_indicators:
                            g.add((indicator_id, SCHEMA.unitText, Literal("grams")))
                            g.add((indicator_id, SCHEMA.quantity, Literal(indicator_value, datatype=XSD.float)))
                        else:
                            g.add((indicator_id, SCHEMA.quantity, Literal(indicator_value, datatype=XSD.string)))

                        # Aggiungi la relazione tra la ricetta e l'indicatore
                        g.add((recipe_id, SCHEMA.NutritionInformation, indicator_id))


    # Relazioni di alternativa tra ricette, è ingrediente, ha ingrediente
    for idx1, row1 in df.iterrows():
        for idx2, row2 in df.iterrows():
            if idx1 != idx2 and pd.notna(row1['product_name']) and row1['product_name'].strip() != "" and pd.notna(row2['product_name']) and row2['product_name'].strip() != "":

                # Relazione di ricette alternative
                if pd.notna(row1['generic_name']) and pd.notna(row2['generic_name']) and row1['generic_name'] == row2['generic_name']:
                    g.add((URIRef(EX[f"recipe_{idx1}"]), SCHEMA.isSimilarTo, URIRef(EX[f"recipe_{idx2}"])))

                # Relazione di ha ingrediente / è ingrediente
                if pd.notna(row2['ingredients_text']) and row1['generic_name'] in str(row2['ingredients_text']).split(', '):
                    g.add((URIRef(EX[f"recipe_{idx1}"]), SCHEMA.isPartOf, URIRef(EX[f"recipe_{idx2}"])))
                    g.add((URIRef(EX[f"recipe_{idx2}"]), SCHEMA.hasPart, URIRef(EX[f"recipe_{idx1}"])))

    # Salva il grafo RDF in formato Turtle
    g.serialize(destination=file_output, format="turtle")
    print(f"File generato con successo: {file_output}")

