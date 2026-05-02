# Computer Vision and NLP Student Project Assignment Guide

Every project must clearly show:

1. What problem you are solving.
2. Which dataset you used.
3. How you cleaned or prepared the data.
4. Which model you trained.
5. How well the model performed.
6. What mistakes the model made.

## Simple Project Approach

Use this same approach for any dataset.

1. Select one dataset.
2. Write the problem statement in 2 to 3 lines.
3. Load the dataset.
4. Show a few examples from the dataset.
5. Clean and preprocess the data.
6. Split the data into train and test sets.
7. Train a simple baseline model.
8. Improve the model if possible.
9. Evaluate using accuracy, precision, recall, F1-score, or task-specific metrics.
10. Write the conclusion.

## Computer Vision Dataset Ideas

| Dataset | Problem Statement | Simple Approach | Level | Link |
|---|---|---|---|---|
| MNIST | Classify handwritten digits from 0 to 9. | Normalize images, train a simple CNN, check accuracy and confusion matrix. | Easy | [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) |
| Fashion-MNIST | Classify clothing items such as shirts, shoes, bags, and trousers. | Normalize images, train CNN, compare correct and wrong predictions. | Easy | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |
| EMNIST | Classify handwritten letters or digits. | Use a CNN like MNIST, then analyze which characters are confusing. | Easy | [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) |
| CIFAR-10 | Classify small color images into 10 object classes. | Train CNN, add data augmentation, compare accuracy before and after improvement. | Easy to Medium | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
| SVHN | Recognize digits from real street house number images. | Resize and normalize images, train CNN, compare with MNIST result. | Medium | [SVHN](http://ufldl.stanford.edu/housenumbers/) |
| Cats vs Dogs | Classify an image as cat or dog. | Train a CNN or use transfer learning with MobileNet/ResNet. | Easy to Medium | [Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765) |
| TF Flowers | Classify flower images into flower categories. | Use image folder loading, resize images, train CNN or transfer learning model. | Easy to Medium | [TF Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) |
| Oxford Flowers 102 | Classify flower images into 102 flower classes. | Use transfer learning because there are many classes and fewer images per class. | Medium | [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| Oxford-IIIT Pet | Classify pet breed from cat and dog images. | Use transfer learning, show sample images, evaluate breed classification. | Medium | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) |
| Stanford Dogs | Classify dog breed from dog images. | Use transfer learning, compare top wrong breed predictions. | Medium | [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) |
| PlantVillage | Detect plant disease from leaf images. | Train CNN or transfer learning model, report which diseases are confused. | Medium | [PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) |
| EuroSAT | Classify satellite images into land-use categories. | Train CNN or transfer learning model, explain real-world use in agriculture or urban planning. | Medium | [EuroSAT](https://github.com/phelber/eurosat) |
| Food-101 | Classify food images into dish categories. | Use transfer learning, test on sample food images, discuss difficult classes. | Medium | [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) |
| Caltech-101 | Classify common objects into many object categories. | Resize images, use transfer learning, evaluate top confused classes. | Medium | [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) |
| PASCAL VOC 2012 | Detect or segment objects in real images. | Use a pre-trained object detection model, test on sample images, calculate detection results. | Advanced | [PASCAL VOC 2012](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/) |
| COCO | Detect objects, segment objects, or generate image captions. | Use pre-trained YOLO/Faster R-CNN or image captioning model; keep scope small. | Advanced | [COCO](https://cocodataset.org/#home) |

## NLP Dataset Ideas

| Dataset | Problem Statement | Simple Approach | Level | Link |
|---|---|---|---|---|
| SMS Spam Collection | Classify SMS messages as spam or not spam. | Clean text, use TF-IDF, train Naive Bayes or Logistic Regression. | Easy | [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| IMDB Movie Reviews | Classify movie reviews as positive or negative. | Use TF-IDF first, then try LSTM or BERT if possible. | Easy to Medium | [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) |
| AG News | Classify news articles into topic categories. | Use TF-IDF and Logistic Regression, then show confusion matrix. | Easy to Medium | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) |
| SST-2 | Classify short sentences as positive or negative sentiment. | Use a simple classifier or fine-tune a small transformer. | Medium | [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) |
| Emotion | Classify text into emotions such as joy, sadness, anger, fear, love, or surprise. | Clean text, vectorize using TF-IDF, train classification model. | Easy to Medium | [Emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| TREC | Classify questions by question type. | Train a model to identify whether a question asks about person, place, number, etc. | Easy to Medium | [TREC](https://huggingface.co/datasets/CogComp/trec) |
| Banking77 | Classify customer banking queries into intent categories. | Use TF-IDF or sentence embeddings, train intent classifier. | Medium | [Banking77](https://huggingface.co/datasets/PolyAI/banking77) |
| Yelp Polarity | Classify restaurant/business reviews as positive or negative. | Use a sample of the dataset, train sentiment classifier, evaluate F1-score. | Medium | [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) |
| Amazon Polarity | Classify Amazon product reviews as positive or negative. | Use a smaller sample, train sentiment model, compare with IMDB. | Medium | [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) |
| DBpedia-14 | Classify short encyclopedia text into topic categories. | Use title and content text, train topic classifier. | Medium | [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) |
| TweetEval Sentiment | Classify tweets by sentiment. | Clean mentions/hashtags carefully, train sentiment classifier. | Medium | [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) |
| CoNLL-2003 | Identify names of people, locations, and organizations in text. | Build a Named Entity Recognition model using token labels. | Advanced | [CoNLL-2003](https://www.tensorflow.org/datasets/catalog/conll2003) |
| SQuAD 2.0 | Answer questions from a paragraph. | Use a pre-trained question-answering model and evaluate answers. | Advanced | [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) |
| WikiText-2 | Train a model to predict the next word in text. | Build a small language model and evaluate perplexity. | Advanced | [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) |

## Recommended Easy Projects

For beginners, choose one of these:

### Computer Vision

1. MNIST digit classification
2. Fashion-MNIST clothing classification
3. CIFAR-10 image classification
4. Cats vs Dogs classification
5. TF Flowers classification

### NLP

1. SMS spam detection
2. IMDB sentiment analysis
3. AG News topic classification
4. Emotion classification
5. TREC question classification

## Example Problem Statements

### Computer Vision Example

```text
The goal of this project is to classify images from the CIFAR-10 dataset into 10 categories such as airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck.
```

### NLP Example

```text
The goal of this project is to classify SMS messages as spam or not spam using text preprocessing and machine learning.
```


### For Computer Vision Projects

1. Show sample images.
2. Count images in each class.
3. Resize images.
4. Normalize pixel values.
5. Train a CNN or transfer learning model.
6. Show accuracy and confusion matrix.
7. Show 5 wrong predictions and explain them.

### For NLP Projects

1. Show sample text examples.
2. Count labels in each class.
3. Clean the text.
4. Convert text into numbers using Bag of Words, TF-IDF, embeddings, or tokenizer.
5. Train a classifier.
6. Show accuracy, precision, recall, and F1-score.
7. Show 5 wrong predictions and explain them.

## Suggested Models

| Project Type | Simple Model | Improved Model |
|---|---|---|
| Image classification | Simple CNN | Transfer learning with MobileNet, ResNet, or EfficientNet |
| Object detection | Pre-trained YOLO | Fine-tuned YOLO or Faster R-CNN |
| Text classification | Naive Bayes or Logistic Regression | LSTM, GRU, or BERT |
| Sentiment analysis | TF-IDF + Logistic Regression | Fine-tuned transformer |
| Named Entity Recognition | Rule-based or BiLSTM | Transformer token classification |
| Question answering | Pre-trained QA model | Fine-tuned QA model |

## Evaluation Metrics

| Task | Metrics |
|---|---|
| Image classification | Accuracy, precision, recall, F1-score, confusion matrix |
| Text classification | Accuracy, precision, recall, F1-score, confusion matrix |
| Imbalanced classification | Precision, recall, F1-score |
| Object detection | IoU, precision, recall, mAP |
| Segmentation | IoU, Dice score |
| Question answering | Exact Match, F1-score |

## Final Report Format

Students should submit a simple report or notebook with these sections:

1. Project title
2. Problem statement
3. Dataset used
4. Dataset examples
5. Preprocessing steps
6. Model used
7. Results
8. Error analysis
9. Conclusion
10. References
