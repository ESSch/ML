import torch
from transformers import pipeline#, ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor, FSMTForConditionalGeneration, FSMTTokenizer

# pipeline2 = pipeline(
#     task="text2text-generation",
#     model="google-t5/t5-base",
#     dtype=torch.float16,
#     device=0
# );


pipeline3 = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    dtype=torch.float16,
    device=0
)

pipeline4 = pipeline(
    task="text-generation",
    model="facebook/wmt19-en-ru",
    dtype=torch.float16,
    device=0
);

def main(url = "https://i.pinimg.com/736x/01/f0/d2/01f0d2329d42c41e9b5cf315665783fd.jpg"):
    result3 = pipeline3(url);
    print("Image to text: ", result3);
    # [{'label': 'lynx, catamount', 'score': 0.43460196256637573}, {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'score': 0.03484790772199631}, {'label': 'snow leopard, ounce, Panthera uncia', 'score': 0.032355185598134995}, {'label': 'Egyptian cat', 'score': 0.023950593546032906}, {'label': 'tabby, tabby cat', 'score': 0.02285381592810154}]
    detected = result3[0]["label"];
    print("Detected: ", detected);
    result4 = pipeline4(f"Translate English to Russina: {detected}");
    print("Translate EN to RU: ", result4[0]["generated_text"]);