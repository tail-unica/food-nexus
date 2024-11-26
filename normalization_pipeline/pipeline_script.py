"""
File che contiene script e dati necessari per la normalizzazione
dei nomi degli ingredienti e la definizione del modello di traduzione
"""


import torch  # type: ignore
import ollama  # type: ignore
import csv
import re
import pint  # type: ignore
import string
import nltk  # type: ignore
from nltk import pos_tag, word_tokenize  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from spellchecker import SpellChecker  # type: ignore
import unicodedata
from pint import UnitRegistry, UndefinedUnitError  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore


def crea_modello_traduzione() -> None:
    """
    Chiamando questa funzione crea un modello di traduzione
    basato su qwen2.5:32b e usabile su ollama

    :return: None
    """

    # Definizione del system prompt del modello
    modelfile = """
    FROM qwen2.5:32b
    SYSTEM You are a highly skilled linguist with a specific task: I will provide you with a single string of input related to food names, ingredients, recipes, or culinary terms. \
    Your objective is to determine the language of the input string. If the string is in English, respond with "eng." If the string is not in English, translate it into English. \
    Please adhere to the following guidelines: \
    - Do not add any comments, explanations, or modifications to your response. \
    - Always respond with either "eng" or the English translation of the provided text. \
    - Do not remove any punctuation mark, for example (",", ".", ";", ":", "!", "?", "(", ")", "\"") in your response. \
    Here are some examples for clarity: \
    - "pane al mais" -> "corn bread" \
    - "apple" -> "eng" \
    - "frutta" -> "fruit" \
    - "cibo" -> "food" \
    - "birne" -> "pear" \
    - "arroz con pollo y verduras frescas" -> "rice with chicken and fresh vegetables" \
    - "bánh mì với thịt và rau củ" -> "sandwich with meat and vegetables" \
    - "パスタとトマトソース" -> "pasta with tomato sauce" \
    - "gnocchi di patate con burro e salvia" -> "potato dumplings with butter and sage" \
    - "choucroute garnie avec des saucisses" -> "sauerkraut with sausages" \
    - "paella de mariscos y arroz amarillo" -> "seafood paella with yellow rice" \
    - "fish and chips" -> "eng" \
    - "tonno! ☺️ pizza bio l rustica 1kg (2x500g) - con tonno agli oli evo, una vera bontà" -> "tuna! ☺️ pizza bio l rustic 1kg (2x500g) - with tuna in evo oils, a true delight" \
    - "cioccolato 🍫 extra fondente - 85% cacao (con aroma di vaniglia naturale)" -> "extra dark chocolate 85 percent cocoa with natural vanilla aroma" \
    - "queso manchego curado 🧀 - ideal para tapas o gratinar" -> "manchego cheese cured ideal for tapas or gratinating" \
    - "bánh cuốn nhân thịt 🥢 - món ăn sáng Việt Nam thơm ngon" -> "bánh cuốn with meat a delicious Vietnamese breakfast dish" \
    - "寿司 🍣 - 新鮮な魚で作られた最高の日本料理" -> "sushi the finest Japanese dish made with fresh fish" \
    - "pomodoro rosso bio 🍅 (1kg) - perfetto per salse fatte in casa" -> "red organic tomato 1kg perfect for homemade sauces" \
    Begin processing the input now. \
    PARAMETER temperature 0
    PARAMETER top_p 0.8
    PARAMETER top_k 1
    """

    # Crea il modello con il system prompt definito e con nome "translation_expert"
    ollama.create(model="translation_expert", modelfile=modelfile)


def translate_to_english_test(text: str) -> str:
    """
    funzione che traduce una stringa in inglese

    :param text: testo da tradurre
    :return: risposta del modello di traduzione
    """
    response = ollama.generate(model="translation_expert", prompt=text)
    print(text, response["response"])
    return response["response"]


# Device su cui si svolgeranno le operazioni
device = "cuda" if torch.cuda.is_available() else "cpu"


# Inizializza il lemmatizzatore, il correttore ortografico, il registro delle unità
lemmatizer = WordNetLemmatizer()
spell_checker = SpellChecker()
ureg = pint.UnitRegistry()

# Definizioni personalizzate di unità non standard
ureg.define("tbsp = 15 * milliliter")
ureg.define("tsp = 5 * milliliter")
ureg.define("cup = 240 * milliliter")
ureg.define("pint = 473.176 * milliliter")
ureg.define("quart = 946.353 * milliliter")
ureg.define("ounce = 28.3495 * gram")
ureg.define("fluid_ounce = 29.5735 * milliliter")


# Lista di suffissi e prefissi da rimuovere
suffixes_and_prefixes = [
    "non",
    "pre",
    "post",
    "anti",
    "bio",
    "extra",
    "over",
    "under",
    "less",
    "full",
    "ate",
    "&quot;",
    "quot",
]


# Dizionario di sigle e valori associati
abbreviations = {
    "evo": "extra virgin olive oil",
    "bbq": "barbecue",
    "qt": "quart",
    "pkg": "package",
    "deg": "degree",
    "gf": "gluten-free",
    "df": "dairy-free",
    "vgn": "vegan",
    "froz": "frozen",
    "rt": "room temperature",
    "bt": "boiling temperature",
    "bld": "boiled",
    "whl": "whole",
    "xt": "extra thick",
    "pwdr": "powder",
    "bkng": "baking",
    "med": "medium",
    "lg": "large",
    "sm": "small",
    "xlg": "extra large",
}

unit_conversion_map = {
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "kg": "gram",
    "kilogram": "gram",
    "kilograms": "gram",
    "mg": "gram",
    "ounce": "gram",
    "ounces": "gram",
    "ml": "milliliter",
    "milliliter": "milliliter",
    "milliliters": "milliliter",
    "l": "milliliter",
    "liter": "milliliter",
    "liters": "milliliter",
    "tbsp": "milliliter",
    "tablespoon": "milliliter",
    "tablespoons": "milliliter",
    "tsp": "milliliter",
    "teaspoon": "milliliter",
    "teaspoons": "milliliter",
    "oz": "milliliter",
    "fluid_ounce": "milliliter",
    "fluid_ounces": "milliliter",
    "cup": "milliliter",
    "cups": "milliliter",
    "pint": "milliliter",
    "pints": "milliliter",
    "quart": "milliliter",
    "quarts": "milliliter",
}


# Elenco delle unità di misura da rimuovere
units_to_remove = ["gram", "milliliters"]


def load_adjectives_to_keep_words(
    input_file="../csv_file/aggettivi_FoodKg.csv",
) -> set[str]:
    """
    Funzione per caricare gli aggettivi unici in `keep_words`

    :param input_file: percorso del file CSV

    :return: un set di stringhe contenenti gli aggettivi unici di FoodKG
    """

    # Inizializza il set di stringhe
    keep_words = set()
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        # Aggiungi ogni aggettivo trovato nel file a `keep_words`
        for row in reader:
            adjective = row["adjective"].strip().lower()
            keep_words.add(adjective)
    return keep_words


# lista parole da non eliminare in aggettivi e avverbi
keep_words = load_adjectives_to_keep_words()


def load_brands_from_csv(
    file_path="../csv_file/brands_filtered.csv",
) -> list[str]:
    """
    Funzione per caricare i nomi dei brand da eliminare dal file CSV apposito

    :param file_path: percorso del file CSV

    :return: una lista di stringhe contenenti i nomi dei brand da eliminare
    """
    # Inizializza la lista di stringhe
    brands = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Controlla se la riga non è vuota
            if row:
                # Aggiunge il nome del brand rimuovendo spazi extra
                brands.append(row[0].strip())

    return brands


"""
li ordino perche cosi se un nome di un brand è compreso in un altro, rimuove il nome piu lungo
ad esempio "rio" e "rio mare" sono entrambi brand, applicando la rimozione del brand al prodotto
"tonno rio mare" rimuoverebbe "rio" e poi e basta senza rimuoveere mare se non fossero ordinati
"""
# Lista di brand da rimuovere
brands = sorted(load_brands_from_csv(), key=len, reverse=True)


def remove_text_in_parentheses(text: str) -> str:
    """
    Funzione per rimuovere il testo tra parentesi

    :param text: il testo in cui è da rimuovere il testo tra parentesi
    :return: il testo con rimosso il testo tra parentesi
    """
    return re.sub(r"\(.*?\)", "", text).strip()


def remove_text_after_comma_or_colon(text: str) -> str:
    """
    Rimuove il testo dopo la prima virgola o dopo il carattere ":".

    :param text: Il testo dal quale rimuovere la parte dopo la prima virgola o il carattere ":".
    :return: Il testo con la parte successiva alla prima virgola o al carattere ":" rimossa.
    """
    return re.split(r"[,:]", text, maxsplit=1)[0].strip()


# Funzione per rimuovere tutta la punteggiatura rimanente
def remove_or_replace_punctuation(text: str) -> str:
    """
    Rimuove o sostituisce la punteggiatura nel testo.

    Sostituisce la punteggiatura tra due parole con uno spazio e rimuove la punteggiatura rimanente.

    :param text: Il testo da cui rimuovere o sostituire la punteggiatura.
    :return: Il testo con la punteggiatura rimanente rimossa o sostituita.
    """
    # Sostituisci la punteggiatura tra due caratteri con uno spazio
    text = re.sub(r"(?<=\w)[^\w\s](?=\w)", " ", text)

    # Rimuovi qualsiasi altra punteggiatura che non sia già stata sostituita
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def remove_numeric_values(text: str) -> str:
    """
    Rimuove i valori numerici nel testo, inclusi quelli con punto o virgola decimale.

    :param text: Il testo da cui rimuovere i valori numerici.
    :return: Il testo senza valori numerici.
    """
    # Rimuove numeri interi e numeri decimali con punto o virgola
    return re.sub(r"\b\d+[.,]?\d*\b", "", text).strip()


# Funzione per tradurre in inglese
def translate_to_english(text: str) -> str:
    """
    Traduce il testo in inglese utilizzando il modello di traduzione.

    :param text: Il testo da tradurre in inglese.
    :return: Il testo tradotto in inglese o il testo originale se la traduzione non è necessaria.
    """
    response = ollama.generate(model="translation_expert", prompt=text)
    if response["response"].strip() != "eng":
        return response["response"]
    return text


# Funzione per espandere le abbreviazioni
def replace_abbreviations(text: str) -> str:
    """
    Sostituisce le abbreviazioni nel testo con la loro forma estesa.

    :param text: Il testo contenente abbreviazioni da sostituire.
    :return: Il testo con le abbreviazioni sostituite dalla forma estesa.
    """
    for key, value in abbreviations.items():
        text = re.sub(
            r"\b" + re.escape(key) + r"\b", value, text, flags=re.IGNORECASE
        )
    return text


# Funzione per rimuovere suffissi e prefissi
def remove_suffixes_prefixes(text: str) -> str:
    """
    Rimuove i suffissi e prefissi da parole autonome nel testo.

    :param text: Il testo da cui rimuovere suffissi e prefissi.
    :return: Il testo con suffissi e prefissi rimossi.
    """
    for item in suffixes_and_prefixes:
        text = re.sub(r"\b" + re.escape(item) + r"\b", "", text)

    return re.sub(r"\s+", " ", text).strip()


# Funzione per rimuovere i nomi di brand
def remove_brands(text: str) -> str:
    """
    Rimuove i nomi di brand dal testo.

    :param text: Il testo da cui rimuovere i brand.
    :return: Il testo con i nomi dei brand rimossi.
    """
    for brand in brands:
        text = re.sub(
            r"\b" + re.escape(brand) + r"\b", "", text, flags=re.IGNORECASE
        )

    return re.sub(r"\s+", " ", text).strip()


# Funzione per convertire il testo in minuscolo
def convert_to_lowercase(text: str) -> str:
    """
    Converte il testo in minuscolo.

    :param text: Il testo da convertire in minuscolo.
    :return: Il testo convertito in minuscolo.
    """
    return text.lower()


# Funzione per normalizzare i caratteri speciali
def normalize_special_characters(text: str) -> str:
    """
    Normalizza i caratteri speciali rimuovendo accenti e caratteri non ASCII.

    :param text: Il testo da normalizzare.
    :return: Il testo con caratteri speciali normalizzati.
    """

    # Normalizza i caratteri Unicode (NFKD rimuove gli accenti)
    normalized_text = unicodedata.normalize("NFKD", text)

    # Rimuovi e caratteri non ASCII
    return "".join(
        c
        for c in normalized_text
        if c in string.ascii_letters + string.digits + string.whitespace
    )


# Funzione per ordinare le parole in ordine alfabetico
def sort_words_alphabetically(text: str) -> str:
    """
    Ordina le parole nel testo in ordine alfabetico.

    :param text: Il testo da ordinare.
    :return: Il testo con le parole ordinate alfabeticamente.
    """
    words = text.split()
    sorted_words = sorted(words)
    return " ".join(sorted_words)


def normalize_quantities(text: str) -> str:
    """
    Normalizza le quantità nel testo, gestendo anche unità separate da spazi.

    :param text: Il testo contenente le quantità da normalizzare.
    :return: Il testo con le quantità normalizzate.
    """
    # Usa un'espressione regolare per trovare pattern di tipo "numero [spazio opzionale] unità"
    pattern = r"(\d+(\.\d+)?)\s*([a-zA-Z_]+)"
    matches = re.finditer(pattern, text)

    for match in matches:
        number = match.group(1)  # Numero trovato
        unit = match.group(
            3
        ).lower()  # Unità trovata (in minuscolo per uniformità)
        original_quantity = match.group(
            0
        )  # Il match originale, con eventuali spazi

        # Mappa l'unità alla sua forma standard (se esiste nella mappa)
        if unit in unit_conversion_map:
            try:
                standardized_unit = unit_conversion_map[unit]
                # Creazione della quantità con unità standard
                quantity = ureg.Quantity(float(number), unit)

                # Conversione all'unità base (gram o milliliter)
                normalized_quantity = quantity.to(standardized_unit)

                # Formatta il risultato in modo chiaro (arrotondato a due decimali)
                normalized_text = (
                    f"{normalized_quantity.magnitude:.2f} {standardized_unit}"
                )

                # Sostituisce l'originale con la quantità normalizzata
                text = text.replace(original_quantity, normalized_text)
            except (
                ureg.UndefinedUnitError,
                ureg.DimensionalityError,
                ValueError,
            ):
                # Se non è una quantità valida o c'è un'incompatibilità, continua senza modificare
                continue

    return text


# Funzione per rimuovere le stopwords
def remove_stopwords(text: str) -> str:
    """
    Rimuove le stopwords dal testo.

    :param text: Il testo da cui rimuovere le stopwords.
    :return: Il testo con le stopwords rimosse.
    """
    # Carica le stopwords della lingua inglese
    stop_words = set(stopwords.words("english"))
    # Tokenizza il testo
    words = word_tokenize(text)
    # Rimuove le stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


# Funzione per convertire il testo al singolare
def lemmatize_text(text: str) -> str:
    """
    Lemmatizza il testo, riducendo le parole alla loro forma base.

    :param text: Il testo da lemmatizzare.
    :return: Il testo lemmatizzato.
    """
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


# Funzione per rimuovere le unità di misura
def remove_units(text: str) -> str:
    """
    Rimuove le unità di misura dal testo.

    :param text: Il testo da cui rimuovere le unità di misura.
    :return: Il testo senza le unità di misura.
    """
    words = text.split()
    cleaned_words = [word for word in words if word not in units_to_remove]
    return " ".join(cleaned_words)


# Funzione per rimuovere verbi, avverbi e aggettivi non inclusi nella lista consentita
def remove_unwanted_pos(text: str) -> str:
    """
    Rimuove verbi, avverbi e aggettivi non inclusi nella lista consentita.

    :param text: Il testo da cui rimuovere i POS indesiderati.
    :return: Il testo con i POS indesiderati rimossi.
    """
    words = word_tokenize(text)

    # ho tolto 'RB' perche toglie troppe informazioni, è da valutare se rimetterlo o meno
    # 'JJ' = aggettivi, 'RB' = avverbi, 'VB' = verbi, 'NNP' = nomi propri
    filtered_words = [
        word
        for word, pos in pos_tag(words)
        if pos not in ("VB", "JJ", "NNP") or word in keep_words
    ]
    return " ".join(filtered_words)


# Funzione che rimuove le parole duplicate
def remove_duplicate_words(text: str) -> str:
    """
    Rimuove le parole duplicate nel testo.

    :param text: Il testo da cui rimuovere le parole duplicate.
    :return: Il testo senza parole duplicate.
    """
    words = word_tokenize(text)
    filtered_words = list(set(words))
    return " ".join(filtered_words)


def remove_lenght1_words(line):
    """
    Rimuove tutte le parole di lunghezza 1 da una stringa.

    Args:
        line (str): La stringa di input.

    Returns:
        str: La stringa con le parole di lunghezza 1 rimosse.
    """
    return " ".join(word for word in line.split() if len(word) > 1)


def pipeline(
    input_file, output_file, mostra_tutto=False, mostra_qualcosa=False
) -> None:
    """
    Funzione per eseguire il pipeline di normalizzazione

    :param input_file: percorso del file da normalizzare
    :param output_file: percorso del file di output
    :param mostra_tutto: se True, mostra tutte le fasi intermedie di normalizzazione
    :param mostra_qualcosa: se True, mostra alcune fasi  di normalizzazione
    """

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:

            # Testo originale
            if mostra_tutto or mostra_qualcosa:
                print("originale:".ljust(40), line, end="")

            # Converti in minuscolo
            line = convert_to_lowercase(line)
            if mostra_tutto:
                print("minuscolo:".ljust(40), line)

            # Rimuovi i nomi di brand
            line = remove_brands(line)
            if mostra_tutto:
                print("brand rimosso:".ljust(40), line)

            # Traduci il testo in inglese
            line = translate_to_english(line)
            if mostra_tutto:
                print("traduzione testo in inglese:".ljust(40), line)

            # Converti in minuscolo
            line = convert_to_lowercase(line)
            if mostra_tutto:
                print("minuscolo:".ljust(40), line)

            # Espansione sigle
            line = replace_abbreviations(line)
            if mostra_tutto:
                print("sigle espanse:".ljust(40), line)

            # Rimuovi verbi, avverbi e aggettivi non inclusi nella lista
            line = remove_unwanted_pos(line)
            if mostra_tutto:
                print("rimozione aggettivi:".ljust(40), line)

            # Rimuovi suffissi e prefissi
            line = remove_suffixes_prefixes(line)
            if mostra_tutto:
                print("rimozione suffissi e prefissi:".ljust(40), line)

            # Rimuovi testo tra parentesi
            line = remove_text_in_parentheses(line)
            if mostra_tutto:
                print("rimozione testo parentesi:".ljust(40), line)

            # Rimuovi testo dopo la prima virgola o ":"
            line = remove_text_after_comma_or_colon(line)
            if mostra_tutto:
                print("rimozione testo dopo virgola:".ljust(40), line)

            # Rimuovi punteggiatura rimanente
            line = remove_or_replace_punctuation(line)
            if mostra_tutto:
                print("rimozione punteggiatura:".ljust(40), line)

            # Rimuovi le stopwords
            line = remove_stopwords(line)
            if mostra_tutto:
                print("rimozione stopwords:".ljust(40), line)

            # Normalizza le quantità
            line = normalize_quantities(line)
            if mostra_tutto:
                print("normalizza quantità:".ljust(40), line)

            # Elimina le misure
            line = remove_units(line)
            if mostra_tutto:
                print("rimozione unità di misura:".ljust(40), line)

            # Rimuovi valori numerici
            line = remove_numeric_values(line)
            if mostra_tutto:
                print("rimozione valori numerici:".ljust(40), line)

            # Converti il testo al singolare
            line = lemmatize_text(line)
            if mostra_tutto:
                print("conversione testo al singolare:".ljust(40), line)

            # Normalizza i caratteri speciali
            line = normalize_special_characters(line)
            if mostra_tutto:
                print("normalizza caratteri speciali:".ljust(40), line)

            # Rimuovi le parole duplicate
            line = remove_duplicate_words(line)
            if mostra_tutto:
                print("rimozione parole duplicate:".ljust(40), line)

            # Rimuovi caratteri singoli
            line = remove_lenght1_words(line)
            if mostra_tutto:
                print("rimozione parole lunghe 1 carattere:".ljust(40), line)

            # Ordina le parole in ordine alfabetico
            line = sort_words_alphabetically(line)
            if mostra_tutto:
                print("ordinamento alfabetico:".ljust(40), line)

            if mostra_qualcosa or mostra_tutto:
                print("normalizzazione completa:".ljust(40), line, "\n")
            outfile.write(line + "\n")

    print("Normalizzazione completa")
