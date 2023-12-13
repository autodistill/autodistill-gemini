import os
from dataclasses import dataclass

import requests
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")


@dataclass
class Gemini(DetectionBaseModel):
    ontology: CaptionOntology
    api_key: str
    gcp_region: str
    gcp_project: str

    def __init__(
        self, ontology: CaptionOntology, api_key: str, gcp_region: str, gcp_project: str
    ) -> None:
        self.ontology = ontology
        self.api_key = api_key
        self.gcp_region = gcp_region
        self.gcp_project = gcp_project

    def predict(self, input: str, prompt: str, confidence: int = 0.5) -> sv.Detections:
        payload = {
            "contents": {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": input,
                        }
                    },
                    {"text": prompt},
                ],
            },
            "safety_settings": {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            "generation_config": {
                "temperature": 0.4,
                "topP": 1.0,
                "topK": 32,
                "maxOutputTokens": 2048,
            },
        }

        response = requests.post(
            f"https://{self.gcp_region}-aiplatform.googleapis.com/v1/projects/{self.gcp_project}/locations/{self.gcp_region}/publishers/google/models/gemini-pro-vision:streamGenerateContent",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    #       "candidates": [
    # {
    #   "content": {
    #     "parts": [
    #       {
    #         "text": string
    #       }
    #     ]
    #   },

        if not response.ok:
            raise Exception(response.text)

        response_body = response.json()

        text_response = response_body["candidates"][0]["content"]["parts"][0]["text"]

        prompts = self.ontology.prompts()

        is_in = []

        for prompt in prompts:
            is_in.append(prompt in text_response)

        return sv.Classifications(
            class_ids=self.ontology.class_ids(),
            confidence=[1 if i else 0 for i in is_in],
        )