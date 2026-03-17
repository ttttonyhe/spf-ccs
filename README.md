## Understanding and Preserving Safety in Fine-Tuned LLMs

Our experiments require a minimum of 4 A100/H100 80GB GPUs, as we perform full parameter fine-tuning with a batch size of 16. The experiments can also utilize up to 8 CPU cores and 256GB of RAM.

<br/>

## Demo
A demo that presents safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace, is available at [notebooks/demo.ipynb](notebooks/demo.ipynb).
