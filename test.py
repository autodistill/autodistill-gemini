from autodistill_gemini import Gemini
from autodistill.detection import CaptionOntology
import os

# define an ontology to map class names to our Gemini prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = Gemini(
    ontology=CaptionOntology(
        {
            "Midnights": "midnights",
            "Reputation": "reputation",
        }
    ),
    api_key=os.environ["GCP_KEY"],
    gcp_region="us-central1",
    gcp_project="roboflow-marketing",
)

result = base_model.predict("image.jpeg")

print(result)

# label a folder of images
# base_model.label("./context_images", extension=".jpeg")