<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Gemini Module

This repository contains the code supporting the Gemini base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Gemini](https://deepmind.google/technologies/gemini/), developed by Google, is a multimodal computer vision model that allows you to ask questions about images. You can use Gemini with Autodistill for image classification.

You can combine Gemini with other base models to label regions of an object. For example, you can use Grounding DINO to identify abstract objects (i.e. a vinyl record) then Gemini to classify the object (i.e. say which of five vinyl records the region represents). Read the Autodistill [Combine Models](https://docs.autodistill.com/utilities/combine-models/) guide for more information.

> [!NOTE]
> Using this project will incur billing charges for API calls to the Gemini API.
> Refer to the [Google Cloud pricing](https://cloud.google.com/pricing/) page for more information and to calculate your expected pricing. This package makes one API call per image you want to label.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

## Installation

To use Gemini with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-gemini
```

## Quickstart

```python
from autodistill_gemini import Gemini

# define an ontology to map class names to our Gemini prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = Gemini(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    ),
    gcp_region="us-central1",
    gcp_project="project-name",
)

# run inference on an image
result = base_model.predict("image.jpg")

print(result)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```

## License

This project is licensed under an [MIT license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!