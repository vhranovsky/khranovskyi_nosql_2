from sentence_transformers import SentenceTransformer, models


def prepare_model(model_name: str = "allenai/specter2_base") -> SentenceTransformer:
    # Підготовка моделі SentenceTransformer з нормалізацією ембеддингів.
    # Щоб позбутися цього повідомлення: No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    return model