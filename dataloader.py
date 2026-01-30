"""
DataLoader for KernelBook and KernelBench datasets.

Loads PyTorch code from:
- GPUMODE/KernelBook: 1500 samples (for SFT with existing Triton solutions)
- ScalingIntelligence/KernelBench: 1000 samples (for RL/trace generation)

Usage:
    uv run python dataloader.py
"""

from datasets import load_dataset
from typing import Iterator, Dict, Any, Optional
import random


class KernelDataLoader:
    """
    Unified dataloader for kernel generation datasets.
    """
    
    def __init__(
        self,
        kernelbook_samples: int = 1500,
        kernelbench_samples: int = 1000,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the dataloader.
        
        Args:
            kernelbook_samples: Number of samples to load from KernelBook (max ~4000)
            kernelbench_samples: Number of samples to load from KernelBench (max 270)
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
        """
        self.kernelbook_samples = kernelbook_samples
        self.kernelbench_samples = kernelbench_samples
        self.shuffle = shuffle
        self.seed = seed
        
        random.seed(seed)
        
        self._kernelbook_data = None
        self._kernelbench_data = None
        
    def load_kernelbook(self) -> list[Dict[str, Any]]:
        """
        Load GPUMODE/KernelBook dataset.
        
        Contains (PyTorch, Triton) pairs - useful for SFT.
        
        Returns:
            List of samples with keys: python_code, triton_code, entry_point
        """
        if self._kernelbook_data is not None:
            return self._kernelbook_data
            
        print(f"Loading KernelBook (target: {self.kernelbook_samples} samples)...")
        
        ds = load_dataset("GPUMODE/KernelBook", split="train")
        
        samples = []
        for i, item in enumerate(ds):
            if i >= self.kernelbook_samples:
                break
                
            samples.append({
                "source": "kernelbook",
                "pytorch_code": item["python_code"],
                "triton_code": item.get("triton_code"),  # Has solution!
                "entry_point": item.get("entry_point", "Model"),
                "index": i,
            })
        
        if self.shuffle:
            random.shuffle(samples)
            
        self._kernelbook_data = samples
        print(f"Loaded {len(samples)} KernelBook samples")
        return samples
    
    def load_kernelbench(self) -> list[Dict[str, Any]]:
        """
        Load ScalingIntelligence/KernelBench dataset.
        
        Contains PyTorch code only (no solutions) - ideal for RL/evaluation.
        Levels: level_1 (100), level_2 (100), level_3 (50), level_4 (20)
        
        Returns:
            List of samples with keys: code, level, name, problem_id
        """
        if self._kernelbench_data is not None:
            return self._kernelbench_data
            
        print(f"Loading KernelBench (target: {self.kernelbench_samples} samples)...")
        
        samples = []
        
        # Load from each level
        levels = ["level_1", "level_2", "level_3", "level_4"]
        samples_per_level = self.kernelbench_samples // len(levels)
        
        for level in levels:
            try:
                ds = load_dataset("ScalingIntelligence/KernelBench", split=level)
                
                for i, item in enumerate(ds):
                    if len([s for s in samples if s.get("level") == int(level.split("_")[1])]) >= samples_per_level:
                        break
                        
                    samples.append({
                        "source": "kernelbench",
                        "pytorch_code": item["code"],
                        "triton_code": None,  # No solution!
                        "level": item["level"],
                        "name": item["name"],
                        "problem_id": item["problem_id"],
                    })
            except Exception as e:
                print(f"Warning: Failed to load {level}: {e}")
        
        if self.shuffle:
            random.shuffle(samples)
            
        self._kernelbench_data = samples
        print(f"Loaded {len(samples)} KernelBench samples")
        return samples
    
    def load_all(self) -> list[Dict[str, Any]]:
        """
        Load all data from both sources.
        
        Returns:
            Combined list of samples from both datasets
        """
        kernelbook = self.load_kernelbook()
        kernelbench = self.load_kernelbench()
        
        all_samples = kernelbook + kernelbench
        
        if self.shuffle:
            random.shuffle(all_samples)
            
        print(f"Total samples: {len(all_samples)}")
        return all_samples
    
    def iter_for_trace_generation(self) -> Iterator[Dict[str, Any]]:
        """
        Iterator for trace generation - yields samples one at a time.
        
        Yields:
            Sample dict with pytorch_code and metadata
        """
        all_samples = self.load_all()
        for sample in all_samples:
            yield sample
    
    @staticmethod
    def extract_inputs_from_code(pytorch_code: str) -> Optional[Dict]:
        """
        Extract input shapes from get_inputs() function in PyTorch code.
        
        This is used to create the input_shapes dict for benchmarking.
        
        Args:
            pytorch_code: The PyTorch code string
            
        Returns:
            Dict with input specifications or None if extraction fails
        """
        # For KernelBench, get_inputs() is executed dynamically on GPU
        # So we return None and let Modal handle it
        return None


def main():
    """Test the dataloader."""
    loader = KernelDataLoader(
        kernelbook_samples=10,  # Small for testing
        kernelbench_samples=10,
    )
    
    print("\n=== KernelBook Samples ===")
    kb_samples = loader.load_kernelbook()
    for s in kb_samples[:2]:
        print(f"  Entry: {s.get('entry_point', 'N/A')}")
        print(f"  PyTorch: {s['pytorch_code'][:100]}...")
        print(f"  Has Triton: {s['triton_code'] is not None}")
        print()
    
    print("\n=== KernelBench Samples ===")
    kbench_samples = loader.load_kernelbench()
    for s in kbench_samples[:2]:
        print(f"  Name: {s.get('name', 'N/A')}")
        print(f"  Level: {s.get('level', 'N/A')}")
        print(f"  PyTorch: {s['pytorch_code'][:100]}...")
        print()


if __name__ == "__main__":
    main()
