# AutoDDG
Automated Dataset Description Generation using Large Language Models

This is the code for the paper "AutoDDG: Automated Dataset Description Generation using Large Language Models" submitted to VLDB 2025. The extended version of the paper is available at this repository ([AutoDDG Extended Version](AutoDataDescription_VLDB2025.pdf)) and the following link: 
- <https://arxiv.org/abs/2502.01050>

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

## How to Use

Following are the steps to use the code for the AutoDDG system.

### Initialization of the OpenAI Client
```python
# You need to have your own api key
client = openai.OpenAI(
            api_key = api_key,
            base_url = base_url
        )
model_name = 'gpt-4o-mini'
```

### Context Preparation
```python
from src.generate_description import SemanticProfiler, DatasetDescriptionGenerator, SearchFocusedDescription
from src.generate_topic import DatasetTopicGenerator
from src.utils import get_sample
from src.data_process import dataset_profiler
import pandas as pd

# Load the dataset and sample
csv_file = 'data/your_dataset.csv'
title = 'Your Dataset Title'
original_description = 'Your Dataset Description'
csv_df = pd.read_csv(csv_file)
sample_df, dataset_sample = get_sample(csv_df, sample_size=reduced_sample_size)

# Load the semantic profiler
semantic_profiler = SemanticProfiler(client=client, model_name=model_name)

# Generate the basic and semantic profiles
basic_profile, semantic_profile_part1 = dataset_profiler(csv_df)
semantic_profile_part2 = semantic_profiler.analyze_dataframe(sample_df)
semantic_profile = semantic_profile_part1+'\n'+semantic_profile_part2

# Generate the dataset topic
data_topic_generator = DatasetTopicGenerator(client=client, model_name=model_name)
data_topic = data_topic_generator.generate_topic(title, original_description, dataset_sample)
```

### Dataset Description Generation
```python
# We use the basic and semantic profiles, and the dataset topic to generate the dataset description
description_generator = DescriptionGenerator(client=client, model_name=model_name)
_, description = description_generator.generate_description(
                    dataset_sample=dataset_sample,
                    dataset_profile=basic_profile,
                    use_profile=True,
                    semantic_profile=semantic_profile,
                    use_semantic_profile=True,
                    data_topic=data_topic,
                    use_topic=True
                )

# Generate the search-focused description
sfd_model = SearchFocusedDescription(client=client, model_name=model_name)
_, search_focused_description = sfd_model.expand_description(initial_descriptionription=description, topic=data_topic)
```

### Quality Evaluation
```python
from src.evaluate import GPTEvaluator

# You need to add your api key in src/evaluate.py
llm_evaluator = GPTEvaluator()
gpt_score = llm_evaluator.evaluate(description)
gpt_score_sfd = llm_evaluator.evaluate(search_focused_description)
```
