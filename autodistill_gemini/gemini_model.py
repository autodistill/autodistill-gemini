import os
from dataclasses import dataclass

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
import supervision as sv
from autodistill.helpers import load_image
import numpy as np
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")


@dataclass
class Gemini(DetectionBaseModel):
    ontology: CaptionOntology
    api_key: str
    gcp_region: str
    gcp_project: str

    def __init__(
        self, ontology: CaptionOntology, gcp_region: str, gcp_project: str
    ) -> None:
        self.ontology = ontology
        self.gcp_region = gcp_region
        self.gcp_project = gcp_project

    def predict(
        self, input: str, prompt: str = "", confidence: int = 0.5
    ) -> sv.Detections:
        if not prompt:
            prompt = "Which of the following labels best describes this image?\n"

            for caption in self.ontology.prompts():
                prompt += f"- {caption}\n"

            prompt += "\n"

            prompt += "Only return the exact label."

        vertexai.init(project=self.gcp_project, location=self.gcp_region)

        multimodal_model = GenerativeModel("gemini-pro-vision")

        response = multimodal_model.generate_content(
            [prompt, Image.load_from_file(input)]
        )

        text_response = response.text.strip()

        prompts = self.ontology.prompts()

        is_in = []

        for prompt in prompts:
            is_in.append(prompt in text_response)

        return sv.Classifications(
            class_id=np.array(
                [self.ontology.prompts().index(caption) for caption in prompts]
            ),
            confidence=np.array([1 if i else 0 for i in is_in]),
        )
