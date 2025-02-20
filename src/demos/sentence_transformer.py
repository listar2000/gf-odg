from sentence_transformers import SentenceTransformer, util
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../../configs/models", config_name="sentence_transformer", version_base="1.1")
def main(cfg: DictConfig) -> None:
    print("Config contents:")
    print(OmegaConf.to_yaml(cfg))
    
    model = SentenceTransformer(
        model_name_or_path=cfg.model_name,
        cache_folder=cfg.cache_dir
    )

    # check number of parameters in model
    num_params = 0
    for _, param in model.named_parameters():
        num_params += param.numel() 
    print(f"Number of parameters: {num_params}")
    
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    embeddings = model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}")

    similarities = util.cos_sim(embeddings, embeddings)
    print("\nSimilarity matrix:")
    print(similarities)

if __name__ == "__main__":
    main()