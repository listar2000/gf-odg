from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datasets import load_dataset, Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    path: str
    config_name: Optional[str] = None
    splits: Optional[List[str]] = None
    cache_dir: Optional[str] = None


class DataLoader:
    """A wrapper class for loading various datasets."""

    def __init__(self, config: DictConfig):
        """Initialize the data loader with configuration.

        Args:
            config: Configuration object containing dataset specifications
        """
        self.config = config
        self.datasets: Dict[str, DatasetDict] = {}

    def load_dataset(self, dataset_name: str, config_name: Optional[str] = None) -> Union[Dataset, DatasetDict]:
        """Load a dataset based on the configuration.

        Args:
            dataset_name: Name of the dataset to load (e.g., 'doqa')
            config_name: Specific configuration name for the dataset (e.g., 'cooking' for doqa)

        Returns:
            Loaded dataset or dataset dictionary containing all splits
        """
        if dataset_name not in self.config.datasets:
            raise ValueError(
                f"Dataset {dataset_name} not found in configuration")

        dataset_config = self.config.datasets[dataset_name]

        # If config_name is not provided but configs exist in dataset_config, use the first one
        if not config_name and hasattr(dataset_config, 'configs'):
            config_name = dataset_config.configs[0].name

        # Create a dataset config object
        config_obj = DatasetConfig(
            name=dataset_config.name,
            path=dataset_config.path,
            config_name=config_name,
            cache_dir=dataset_config.get('cache_dir', None)
        )

        # Set the splits if they exist in the config
        if hasattr(dataset_config, 'configs'):
            for cfg in dataset_config.configs:
                if cfg.name == config_name:
                    config_obj.splits = cfg.splits
                    break

        return self._load_dataset(config_obj)

    def _load_dataset(self, config: DatasetConfig) -> Union[Dataset, DatasetDict]:
        """Internal method to load a dataset using the Hugging Face datasets library.

        Args:
            config: Dataset configuration object

        Returns:
            Loaded dataset or dataset dictionary
        """
        kwargs = {
            "path": config.path,
        }

        if config.config_name:
            kwargs["name"] = config.config_name

        if config.cache_dir:
            kwargs["cache_dir"] = config.cache_dir

        try:
            dataset = load_dataset(**kwargs)
            self.datasets[config.name] = dataset
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {config.name}: {str(e)}")

    def get_dataset(self, dataset_name: str) -> Optional[Union[Dataset, DatasetDict]]:
        """Get a previously loaded dataset.

        Args:
            dataset_name: Name of the dataset to retrieve

        Returns:
            The loaded dataset if it exists, None otherwise
        """
        return self.datasets.get(dataset_name)

    def format_for_instruction_tuning(
        self,
        dataset: Dataset,
        prompt_template: str,
        input_fields: Dict[str, str],
        max_length: Optional[int] = None,
    ) -> Dataset:
        """Format a dataset for instruction tuning.
        
        Args:
            dataset: The dataset to format
            prompt_template: Template string with placeholders for input fields
            input_fields: Mapping of template placeholders to dataset column names
            max_length: Optional maximum length for truncation
            
        Returns:
            Formatted dataset ready for instruction tuning
        """
        def format_example(example):
            # Just use question and answer fields
            formatted_text = prompt_template.format(
                question=example[input_fields["question"]],
                answer=example[input_fields["answer"]]
            )
            return {"text": formatted_text}
        
        # Map and remove all columns except 'text'
        return dataset.map(format_example).remove_columns(
            [col for col in dataset.column_names if col != "text"]
        )


if __name__ == "__main__":
    config = OmegaConf.load("configs/main.yaml")
    data_loader = DataLoader(config)

    dataset = data_loader.load_dataset("career_qa")

    print("Dataset info:")
    print(dataset)

    train_data = dataset['train']
    print("\nFirst example from train set:")
    example = train_data[0]
    print("Role", example['role'])
    print("Question", example['question'])
    print("Answer", example['answer'])
