from __future__ import annotations

from typing import Any

from beartype import beartype

from ..utils import load_prompts


@beartype
class BaseEvaluator:
    """Base class that implements the shared evaluation workflow

    Needs to be inherited through any custom evaluator implementations

    Args:
        client: LLM client instance
        model_name: Model name to use for evaluation
    """

    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model = model_name
        prompts = load_prompts()["evaluation"]
        self._system_message = prompts["system_message"].strip()
        self._evaluation_prompt = prompts["user_prompt"]

    def _build_content(self, description: str) -> str:
        return (
            f"{self._evaluation_prompt}\n"
            f"Description: {description}\n"
            "Evaluation Form (scores ONLY): "
        )

    def evaluate(self, description: str) -> str:
        """
        Evaluate the given description text & Return the raw scoring response from the model

        Args:
            description: Description text

        Returns:
            Evaluation response
        """

        content = self._build_content(description)
        return self._generate(content)

    def _generate(self, content: str) -> str:
        evaluation = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        return evaluation.choices[0].message.content

@beartype
class PreferenceEvaluator(BaseEvaluator):
    """
    Evaluator that compares two descriptions based on researcher utility 
    (LLM Preference).
    """

    def __init__(self, client: Any, model_name: str) -> None:
        super().__init__(client, model_name)
        
        # 1. Define the specific system and user prompts for preference
        # Use the structured prompts we designed previously (Llama 3.1 focused)

        self._system_message = (
            "You are a highly experienced and objective Research Scientist specializing in data evaluation. "
            "Your task is to compare two dataset descriptions and determine which is superior from a utility "
            "and professional perspective. You must be extremely critical and base your judgment ONLY on "
            "the quality of the descriptions provided. You must output your response in a single, complete JSON object. "
            "DO NOT output any text outside the JSON block."
        )

        self._evaluation_prompt_template = (
            "You have been provided with two descriptions (A and B) for a new dataset. "
            "Your goal is to select the description that would be more helpful to a fellow researcher looking to use this dataset.\n\n"
            "[CRITERIA]\n"
            "1. Clarity: Is the description logically structured and easy to understand?\n"
            "2. Completeness: Does it mention all crucial elements (e.g., source, size, domain, purpose, limitations/bias)?\n"
            "3. Researcher Utility: Which one is more professionally-toned, objective, and provides the most relevant detail for a research-oriented audience?\n\n"
            "[DESCRIPTION A]\n{description_a}\n\n"
            "[DESCRIPTION B]\n{description_b}\n\n"
            "[OUTPUT_FORMAT]\n"
            "Respond ONLY with a JSON object containing the following keys:\n"
            "- 'Preference': The letter of the preferred description ('A', 'B', or 'Tie').\n"
            "- 'Score_A': An integer score for Description A (1-5, where 5 is best).\n"
            "- 'Score_B': An integer score for Description B (1-5, where 5 is best).\n"
            "- 'Rationale': A detailed, objective paragraph explaining the choice or the tie, referencing the [CRITERIA] above. Focus on specific flaws or strengths in each description."
        )

    # 2. Override _build_content to accept two descriptions
    def _build_content(self, description_a: str, description_b: str) -> str:
        # Use str.format() to fill the template with both descriptions
        return self._evaluation_prompt_template.format(
            description_a=description_a, 
            description_b=description_b
        )

    # 3. Override evaluate to accept two descriptions
    def evaluate(self, description_a: str, description_b: str) -> str:
        """
        Evaluate the two given description texts & Return the raw JSON response from the model indicating preference.
        """
        content = self._build_content(description_a, description_b)
        # Note: self._generate will use the _system_message defined in __init__
        return self._generate(content)