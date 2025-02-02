# AutoDDG
Automated Dataset Description Generation using Large Language Models

This is the code for the paper "AutoDDG: Automated Dataset Description Generation using Large Language Models" submitted to VLDB 2025. The extended version of the paper is available at this repository ([AutoDDG Extended Version](AutoDataDescription_VLDB2025.pdf)) and the following link: 
- hold

## Requirements
The paper experiments were run using `Python 3.9.9` with the following required packages. They are also listed in the `requirements.txt` file.
- datamart_profiler==0.11
- fastembed==0.4.2
- nltk==3.9.1
- numpy==2.0.2
- openai==1.47.1
- pandas==2.2.3
- rank_bm25==0.2.2
- scikit_learn==1.5.2

You can install the dependencies using `pip`:
```
pip install -r requirements.txt
```

The instructions assume a Unix-like operating system (Linux or MacOS). You may need to adjust the steps for machines running Windows.