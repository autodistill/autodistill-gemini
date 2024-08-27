import os
from dataclasses import dataclass

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from PIL import Image
import supervision as sv
from autodistill.helpers import load_image
import google.generativeai as genai
import numpy as np
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.classification import ClassificationBaseModel

HOME = os.path.expanduser("~")


@dataclass
class GeminiForObjectDetection(DetectionBaseModel):
    ontology: CaptionOntology
    api_key: str
    gcp_region: str
    gcp_project: str
    model: str

    def __init__(
        self, ontology: CaptionOntology, model: str = "gemini-1.5-pro-latest", api_key: str = None
    ) -> None:
        genai.configure(api_key=api_key)
        self.ontology = ontology
        self.model = genai.GenerativeModel(model_name=model)
        
    def predict(
        self, input: str, prompt: str = "", confidence: int = 0.5
    ) -> sv.Detections:
        if not prompt:
            prompt = "Return bounding boxes around every instance of the following labels in the image:\n" + "\n".join(
                self.ontology.prompts()
            ) + """\nReturn in the format {label: [x1, y1, x2, y2]}"""

        response = self.model.generate_content(
            [Image.open(input), prompt]
        )

        # "text": "- [person, 275, 0, 999, 918]\n- [a forklift, 201, 95, 728, 851]\n
        # extract

        text_response = response.text.strip()
        import json
        print(text_response)
        text_as_json = json.loads(text_response)

        detection_bboxes = []
        detection_classes = []

        for detection in text_as_json:
            detection_class = detection

            if detection_class in self.ontology.prompts():
                detection_classes.append(self.ontology.prompts().index(detection_class))
                detection_bboxes.append(text_as_json[detection])

        # detection_bboxes = sv.xyxy_(np.array(detection_bboxes))

        return sv.Detections(
            class_id=np.array(detection_classes),
            xyxy=np.array(detection_bboxes),
            confidence=np.ones(len(detection_classes)),
        )

@dataclass
class GeminiForClassification(ClassificationBaseModel):
    AVAILABLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]
    ontology: CaptionOntology
    api_key: str
    gcp_region: str
    gcp_project: str
    model: str

    def __init__(
        self, ontology: CaptionOntology, gcp_region: str, gcp_project: str, model: str
    ) -> None:
        self.ontology = ontology
        self.gcp_region = gcp_region
        self.gcp_project = gcp_project

        if model in self.AVAILABLE_MODELS:
            self.model = model
        else:
            raise ValueError(f"Choose one of the available models from {self.AVAILABLE_MODELS}")


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

        multimodal_model = GenerativeModel(self.model)

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
