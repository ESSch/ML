import torch
from transformers import pipeline#, ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor, FSMTForConditionalGeneration, FSMTTokenizer

# pipeline2 = pipeline(
#     task="text2text-generation",
#     model="google-t5/t5-base",
#     dtype=torch.float16,
#     device=0
# );
# result2 = pipeline2("translate English to French: The weather is nice today.");
# print("Translate EN to FR: ", result2);

pipeline3 = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    dtype=torch.float16,
    device=0
)
result3 = pipeline3("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg");
print("Image to text: ", result3);
# [{'label': 'lynx, catamount', 'score': 0.43460196256637573}, {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'score': 0.03484790772199631}, {'label': 'snow leopard, ounce, Panthera uncia', 'score': 0.032355185598134995}, {'label': 'Egyptian cat', 'score': 0.023950593546032906}, {'label': 'tabby, tabby cat', 'score': 0.02285381592810154}]
detected = result3[0]["label"];
print("Detected: ", detected);

pipeline4 = pipeline(
    task="text-generation",
    model="facebook/wmt19-en-ru",
    dtype=torch.float16,
    device=0
);
result4 = pipeline4(f"translate English to Russina: {detected}");

# from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
# from PIL import Image
# import requests

# url = "https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/image-classification/test_images/kitten.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384') # https://huggingface.co/google/vit-base-patch16-384
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
# inputs = feature_extractor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# predicted_class_idx = logits.argmax(-1).item()
# detected = model.config.id2label[predicted_class_idx];
# print("Detected: ", detected);

# detected = "The weather is nice today.";
# from transformers import FSMTForConditionalGeneration, FSMTTokenizer
# mname = "facebook/wmt19-en-ru"
# tokenizer = FSMTTokenizer.from_pretrained(mname)
# model = FSMTForConditionalGeneration.from_pretrained(mname)
# input_ids = tokenizer.encode(detected, return_tensors="pt")
# outputs = model.generate(input_ids)
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(decoded)