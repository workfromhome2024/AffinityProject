from django.test import TestCase

import random
from datasets import load_dataset, DatasetDict

def download_and_sample_vla_data(dataset_name="lerobot/svla_so100_stacking", split="test", num_samples=1):
    """
    Downloads a dataset from Hugging Face, selects a specific split, 
    and samples a random subset of examples.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        split (str): The dataset split to use (e.g., "train", "validation", "test").
        num_samples (int): The number of random samples to retrieve.

    Returns:
        list: A list of random data samples (dictionaries).
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split=split)
        
        print(f"Successfully loaded dataset. Total samples in split: {len(dataset)}")

        # Ensure we don't sample more than the total available samples
        if num_samples > len(dataset):
            num_samples = len(dataset)
            print(f"Warning: num_samples adjusted to {num_samples} (maximum available).")

        # Get random indices
        random_indices = random.sample(range(len(dataset)), num_samples)

        # Retrieve the random samples
        random_samples = [dataset[i] for i in random_indices]

        print(f"Retrieved {len(random_samples)} random samples.")
        return random_samples

    except Exception as e:
        print(f"An error occurred during retrieving data from the {dataset_name}: {e}")
        return None
