import torch
from transformers import pipeline

# https://huggingface.co/docs/transformers/model_doc/vit
pipeline3 = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    dtype=torch.float16,
    device=0
)

clip = pipeline(
   task="zero-shot-image-classification",
   model="openai/clip-vit-base-patch32",
   dtype=torch.bfloat16,
   device=0
)

pipeline4 = pipeline(
    task="text-generation",
    model="facebook/wmt19-en-ru",
    dtype=torch.float16,
    device=0
);

# https://docs.langchain.com/oss/javascript/integrations/tools/duckduckgo_search
# https://huggingface.co/IDEA-Research/grounding-dino-base
# https://huggingface.co/keras/retinanet_resnet50_fpn_v2_coco
# https://huggingface.co/segmind/SSD-1B
# https://huggingface.co/models?search=YOLOv12
# https://huggingface.co/models?search=u-net # по идее лучше YOLOv11
def _detection(url):
    # TODO получить полигоны (YOLO)
    # TODO классификация изображения в полегоне (resnet)
    return "";

def validate(url):
    # TODO: проверить, что изображение соответствует запросу
    results = _detection(url);
    return "";

# https://huggingface.co/mradermacher/Quen2-65B-i1-GGUF
# Quen-3B VideoRAM=8GB Latency=3sec
# DINO-2
def getDescription(url):
    results = _detection(url);
    # TODO: получить описание изображения для генерации описания описания товара
    return "";

# распознование документов на русском языке и их валидация
# https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending&search=paddlePaddle1
# https://huggingface.co/URIIT/mns-tesseract
def ocr(url):
    # TODO
    return "";

# TODO: найти товары по описанию
def findText(text):
    # TODO: описк товаров через через RAG
    results = [];
    return results;

# поиск товаров со схожими изображениями
def findImage(url):
    results = _detection(url);
    # TODO: перевести изображение в текст или поиск в RAG-Image
    results = [];
    return results;

def compare(url, label = "dog"):
    labels = [f"a photo of a {label}"];
    result = clip(url, candidate_labels=labels);
    print("Compare: ", result);
    return result;

def detect_image_to_text(url = "https://i.pinimg.com/736x/01/f0/d2/01f0d2329d42c41e9b5cf315665783fd.jpg"):
    result3 = pipeline3(url);
    print("Image to text: ", result3);
    # [{'label': 'lynx, catamount', 'score': 0.43460196256637573}, {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'score': 0.03484790772199631}, {'label': 'snow leopard, ounce, Panthera uncia', 'score': 0.032355185598134995}, {'label': 'Egyptian cat', 'score': 0.023950593546032906}, {'label': 'tabby, tabby cat', 'score': 0.02285381592810154}]
    detected = result3[0]["label"];
    print("Detected: ", detected);
    result4 = pipeline4(f"Translate English to Russina: {detected}");
    print("Translate EN to RU: ", result4[0]["generated_text"]);