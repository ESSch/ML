import torch
import time
from transformers import pipeline

# https://huggingface.co/docs/transformers/model_doc/vit
# pipeline3 = pipeline(
#     task="image-classification",
#     model="google/vit-base-patch16-224",
#     dtype=torch.float16,
#     device=0
# )

# clip = pipeline(
#    task="zero-shot-image-classification",
#    model="openai/clip-vit-base-patch32",
#    dtype=torch.bfloat16,
#    device=0
# )

# pipeline4 = pipeline(
#     task="text-generation",
#     model="facebook/wmt19-en-ru",
#     dtype=torch.float16,
#     device=0
# );

# https://colab.research.google.com/drive/1d46BJslfYXSpzNhN1hR9wZL49nV1BYmP#scrollTo=bUDVF3PAlA9d
# https://huggingface.co/docs/transformers/model_doc/yolos
detector = pipeline(
    task="object-detection",
    model="hustvl/yolos-base",
    dtype=torch.float16, 
    device=0
)

# https://docs.langchain.com/oss/javascript/integrations/tools/duckduckgo_search
# https://huggingface.co/IDEA-Research/grounding-dino-base
# https://huggingface.co/keras/retinanet_resnet50_fpn_v2_coco
# https://huggingface.co/segmind/SSD-1B
# https://huggingface.co/models?search=YOLOv12
# https://huggingface.co/models?search=u-net # по идее лучше YOLOv11
def _detection(url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"):
    start_time = time.time()
    results = detector(url);
    images = [];
    for result in results:
        images.append(result["label"]);
    end_time = time.time();
    elapsed_time = end_time - start_time;
    print(f"Elapsed time: {elapsed_time:.2f} seconds"); # estimated: float16=120.74seconds in colab
    return(images);

def validate(url):
    # TODO: проверить, что изображение соответствует запросу
    results = _detection(url);
    return "";

# https://huggingface.co/mradermacher/Quen2-65B-i1-GGUF
# https://huggingface.co/docs/transformers/model_doc/qwen2_vl
# DINO-2
def getDescription(url):
    results = _detection(url);
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    # TODO: получить описание изображения для генерации описания описания товара
    return "";

# распознование документов на русском языке и их валидация
# https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending&search=paddlePaddle1
# https://huggingface.co/URIIT/mns-tesseract
# https://huggingface.co/docs/transformers/model_doc/trocr
# https://colab.research.google.com/drive/1d46BJslfYXSpzNhN1hR9wZL49nV1BYmP#scrollTo=nZHgUksI5h-m
def OCR(url = "https://habrastorage.org/r/w1560/getpro/habr/post_images/227/beb/221/227beb22178af3b4e99d588e0e96de1a.png"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import requests
    from PIL import Image
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB");
    processor_ru = TrOCRProcessor.from_pretrained("raxtemur/trocr-base-ru")
    model_ru = VisionEncoderDecoderModel.from_pretrained("raxtemur/trocr-base-ru")
    pixel_values_ru = processor_ru(image, return_tensors="pt").pixel_values
    generated_ids_ru = model_ru.generate(pixel_values_ru)
    generated_text_ru = processor_ru.batch_decode(generated_ids_ru, skip_special_tokens=True)[0]
    return generated_text_ru;

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