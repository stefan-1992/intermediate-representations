# Supplementary Material for the Journal Paper:
Fuchs, S., Dimyadi, J., Witbrock, M., and Amor, R. (2024). Improving the semantic parsing of building regulations through intermediate representations. In Advanced Engineering Informatics. Under Review.

### Data Files
- data/results/*: Results for the experiments in Chapter 7. The three runs per setup were averaged in the _combined files.
- data/lrml_ds_v7.csv: LRML Dataset used for establishing Old Baseline in Table 1 (Fuchs et al., 2023a)
- data/lrml_ds_v8.csv: LRML Dataset with evaluation-based cleansing applied - With Reversible IR applied (and partially reverted for the experiments in Section 4.1)

### Experiment Scripts
- egice_experiment_rev_ir.py: Reversible IR experiments.
- egice_experiment_ir.py: Lossy IR experiments.
- egice_experiment_paraphrase.py: Paraphrase as IR experiments.
- egice_experiment_consistency.py: Self Consistency experiments.
- egice_experiment_paraphrase_noir.py: Training with paraphrases without IRs experiment.
- egice_experiment_no_gen.py: Baseline without generation parameters.
- egice_experiment_old_ds.py: Baseline with old dataset.

### Utility Files
- egice_results.ipynb: Generate the _combined result files.
- egice_training.py: Training the T5 model with a custom training loop.
- training_utils.py: Utility functions for the training loop.
- lrml_score.py: Functions to calculate the LRML F1 score
- lrml_utils.py: Additional utility functions for the LRML reversible IR.


### Setup
- requirements.txt: Required packages for the experiments.
- run.ipynb: Jupyter Notebook to run the experiments. 

Steps:
1. Create your virtual environment. Python 3.8 was used for the experiments.
2. Install the required packages with `pip install -r requirements.txt`.
3. Create a .env file with the following content:
```
    WANDB_PROJECT=[YOUR_WANDB_PROJECT]
    WANDB_ENTITY=[YOUR_WANDB_ENTITY]
```
4. Run the experiments with the Jupyter Notebook or the Python scripts.


# Disclaimer
The legal clauses used to train and evaluate the transformer-based semantic parser were extracted from the Acceptable Solutions and Verification Methods for the New Zealand Building Codes (https://www.building.govt.nz/building-code-compliance/). The legal clauses are used for research purposes only and are not intended to be used for any other purpose. The legal clauses are not up-to-date and should not be used for any regulatory or compliance purposes. For alignment purposes, some paragraphs were split into multiple clauses, and some information has been removed. The legal clauses are provided as is and without any warranty.

The LegalRuleML rules are based on Dimyadi et al. (2020) (https://github.com/CAS-HUB/nzbc-lrml). To establish a consistent semantic parsing dataset, some rules were modified, merged, or removed in Fuchs et al. (2022, 2023a,b). Please report any errors in this repository. These errors will be fixed for the next version of the dataset. The LegalRuleML rules are provided as is and without any warranty.

# References

- Dimyadi, J., Fernando, S., Davies, K., & Amor, R. (2020). Computerising the New Zealand building code for automated compliance audit. New Zealand Built Environment Research Symposium (NZBERS).
- Fuchs, S., Witbrock, M., Dimyadi, J., and Amor, R. (2022). Neural semantic parsing of building regulations for compliance checking. In IOP Conference Series: Earth and Environmental Science, volume 1101.
- Fuchs, S., Dimyadi, J., Witbrock, M., and Amor, R. (2023a). Training on digitised building regulations for automated rule extraction. In eWork and eBusiness in Architecture, Engineering and Construction: ECPPM 2022. CRC Press.
- Fuchs, S., Dimyadi, J., Witbrock, M., and Amor, R. (2023b). Improving the semantic parsing of building regulations through intermediate representations. In EG-ICE 2023 Workshop on Intelligent Computing in Engineering, Proceedings.