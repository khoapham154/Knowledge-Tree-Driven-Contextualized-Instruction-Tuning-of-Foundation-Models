# Knowledge-Tree-Driven Contextualized Instruction Tuning for Epilepsy Drug Recommendation

## Abstract
Epilepsy affects over 50 million people worldwide, with antiseizure medications (ASMs) as the primary treatment for seizure control. However, ASM selection remains a "trial and error" process due to the lack of reliable predictors of effectiveness and tolerability. While machine learning approaches have been explored, existing models are limited to predicting outcomes only for ASMs encountered during training and have not leveraged recent biomedical foundation models for this task. This work investigates ASM recommendation using only patient MRI scans and reports. Specifically, we leverage biomedical vision-language foundation models and introduce a novel contextualized instruction-tuning framework that integrates expert-built knowledge trees of MRI entities to enhance their performance. Additionally, while training only on the four most commonly prescribed ASMs, our framework enables generalization to predicting outcomes and effectiveness for unseen ASMs not present during training. We evaluate our instruction-tuning framework on two retrospective epilepsy patient datasets, achieving an average AUC of 71.39 and 63.03 in predicting outcomes for four primary ASMs and three completely unseen ASMs, respectively. Our approach improves the AUC by 5.53 and 3.51 compared to standard report-based instruction tuning for seen and unseen ASMs respectively. 

## Proposed Method

### 1. Construction of Contextualized Instruction Tuning Dataset
We construct our instruction-tuning dataset using the following script:
```bash
python Dataset_Instruction/gpt_4o_clinical.py
```

### 2. Instruction-Tuning of Foundation Models
To perform instruction tuning, execute the following script:
```bash
python Experiment/combined_pipeline.py
```

### 3. Inference
For inference, use the following script:
``` bash
python Inference/inference.py
```

## MRI Entity Distribution

An integral component of TREE-TUNE is the detailed analysis of MRI entity distribution. The project provides visual insights into the organization and annotation of MRI entities: [MRI Entities Distribution](MRI_entities/Entities_distribution.png)

For a complete view of the entity taxonomy, refer to: [Knowledge_MRI_Tree](MRI_entities/Knowledge_MRI_Tree.conf)


