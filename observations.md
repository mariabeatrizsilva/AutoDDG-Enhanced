Updated pipeline and tried to generate, found that current prompts made description too flowery -- 

related_work_extraction:
  default_prompt: |
    You are an expert academic research assistant. Your task is to analyze the provided research paper and extract specific details about a named dataset.
    
    **INSTRUCTIONS:**
    1. Focus solely on the dataset named: **{dataset_name}**.
    2. Provide a concise summary (approx. 3-4 paragraphs, or 500 words maximum) of its **context**, **key characteristics** (e.g., size, source, format, date range), and **how the authors used it** in their research.
    3. Do not include any content from this prompt or any introductory phrases (like "Based on the paper...") in your final answer.
    
    **RESEARCH PAPER TEXT:**
    {paper_text}
  
  system_message: |
    You are an expert academic research assistant specializing in extracting dataset information from research papers.

  related_work_instruction: |
    Furthermore, related research work provides the following context about this dataset:
    {related_profile}
    Based on this information, please add sentence(s) that contextualize the dataset within its research domain and explain its provenance, typical applications, and/or limitations.

produced 

This dataset contains electrocardiogram (ECG) recordings and associated clinical features for patients with heart disease. The data includes variables such as exam_id, age, is_male, nn_predicted_age, timey, normal_ecg, trace_file, patient_id, death, RBBB, LBBB, SB, ST, AF, and 1dAVb. The dataset profile indicates that the data types range from integer to float, with coverage spans for each variable. The semantic profile reveals that the variables represent identification, temporal data, boolean values, measurement values, and general information. This dataset is primarily used in biomedical engineering and cardiology research, specifically for detecting and predicting atrial fibrillation from ECG recordings.

The dataset is part of a larger research study on machine learning algorithms for heart disease diagnosis and treatment. It provides annotated ECG recordings and associated clinical features for patients with atrial fibrillation, allowing researchers to evaluate the performance of different detection and prediction algorithms. The dataset can be used to develop and test new approaches for identifying patients at risk of mortality due to atrial fibrillation.

The dataset is a valuable resource for biomedical engineers, cardiologists, and researchers interested in developing machine learning models for heart disease diagnosis and treatment. It offers a unique opportunity to explore the relationship between ECG recordings and clinical features in predicting atrial fibrillation and mortality risk.

which got score 
Completeness: 8
Conciseness: 6
Readability: 7
