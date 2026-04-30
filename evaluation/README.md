# Evaluation
The evaluationscript can be run by:
```sh
conda run -n tokensmith python evaluation/run_evaluation.py \
  --a-source data/Chapter19--extracted_markdown.md \
  --b-source evaluation/data/Chapter19--updated-extracted_markdown.md \
  --index-prefix evaluation
```

This script includes the following workloads:

1. Build index `A`, using only `data/Chapter19--extracted_markdown.md`
2. Index `B` will consist of all content in `A`, except for one section `19.1  Failure Classification` modified. The extracted markdown file is found at `evaluation/data/Chapter19--updated-extracted_markdown.md`
2. Build index `B` incrementally and save the index snapshot in memory 
3. Build index `B` again with a fresh clean rebuild and save the index snapshot in memory
4. Compare `B_incremental` vs `B_clean`
5. Write all relevant timing and chunk metrics to `evaluation/results/`
