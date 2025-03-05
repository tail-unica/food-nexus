"""
File containing scripts and data necessary for the creation of LLM models
"""


import ollama


def create_food_expert() -> None:
    """
    Calling this function creates a brand filtering model
    based on qwen2.5:32b and usable on Ollama

    :return: None
    """

    with open("../llm_model_file/food_expert.txt", "r") as file:
        model_definition = file.read()

    ollama.create(
        model="food_expert",
        from_="qwen2.5:32b", 
        system=model_definition, 
        parameters={
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 1
        }
    )


def create_attribute_extractor() -> None:
    """
    Calling this function creates a attributes extractor model
    based on qwen2.5:32b and usable on Ollama

    :return: None
    """
    with open("../llm_model_file/attribute_extractor.txt", "r") as file:
        model_definition = file.read()

    ollama.create(
        model="attribute_extractor",
        from_="qwen2.5:32b", 
        system=model_definition, 
        parameters={
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 1
        }
    )


def create_translation_model() -> None:
    """
    Calling this function creates a translation model
    based on qwen2.5:32b and usable on Ollama

    :return: None
    """
    with open("../llm_model_file/translation_expert.txt", "r") as file:
        model_definition = file.read()

    ollama.create(
        model="translation_expert",
        from_="qwen2.5:32b", 
        system=model_definition, 
        parameters={
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 1
        }
    )

    