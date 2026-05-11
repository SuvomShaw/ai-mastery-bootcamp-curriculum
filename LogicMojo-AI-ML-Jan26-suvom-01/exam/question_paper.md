# AI/ML Deep Learning Exam

Duration: 90 minutes  
Total marks: 100

## Instructions

- Attempt all sections.
- For coding, edit only `coding/student.py`.
- Fixed datasets are in `coding/data/`.
- Fixed sample inputs and outputs are in `coding/sample_io.md`.
- Your code should be vectorized where reasonable and should avoid data leakage.
- PyTorch code must run on CPU. Do not assume a GPU is available.
- Write clear, concise answers for descriptive questions.

## Section A: MCQ

Each question carries 1 mark.

1. You are building customer features for a purchase classifier. Which operation most clearly causes target leakage?
   - A. One-hot encoding `event_type`
   - B. Using the target column itself as an input feature
   - C. Filling missing `price` values with the median price
   - D. Counting purchases per `customer_id`

2. In pandas, why is `df.loc[mask].copy()` often safer before adding engineered columns?
   - A. It always makes code faster
   - B. It avoids mutating a view unexpectedly and reduces chained-assignment bugs
   - C. It automatically removes duplicate rows
   - D. It changes all object columns to categorical

3. A CNN receives input shape `(batch, 3, 64, 64)`. A `Conv2d(3, 16, kernel_size=3, stride=2, padding=1)` produces which spatial shape?
   - A. `(16, 31, 31)`
   - B. `(16, 32, 32)`
   - C. `(16, 33, 33)`
   - D. `(3, 32, 32)`

4. Why should `model.train()` be called before a PyTorch training loop?
   - A. It freezes all gradients
   - B. It switches layers like Dropout and BatchNorm into training behavior
   - C. It sends the model to GPU
   - D. It resets all weights

5. `nn.CrossEntropyLoss` in PyTorch expects:
   - A. probabilities after softmax and one-hot labels
   - B. raw logits and integer class labels
   - C. sigmoid outputs and float labels only
   - D. normalized embeddings and cosine labels

6. A binary classifier has 98% accuracy on a dataset where 98% of labels are negative. What is the best immediate concern?
   - A. The model is definitely excellent
   - B. Accuracy may hide poor recall on the positive class
   - C. The learning rate must be too high
   - D. BatchNorm is impossible to use

7. If class counts are `[900, 100]`, why might inverse-frequency sampling help?
   - A. It removes the minority class
   - B. It makes every mini-batch contain only minority samples
   - C. It increases the chance of seeing minority-class samples during training
   - D. It changes the target labels into probabilities

8. In a CNN, increasing receptive field usually helps because:
   - A. the model can use larger context from the image
   - B. it removes the need for nonlinearities
   - C. it guarantees no overfitting
   - D. it makes all filters identical

9. Which augmentation is usually unsafe for digit classification if labels must remain exact?
   - A. Small random translation
   - B. Mild brightness jitter
   - C. Horizontal flip for asymmetric digits
   - D. Small random rotation

10. Why is `optimizer.zero_grad()` normally called each iteration?
    - A. PyTorch accumulates gradients by default
    - B. It resets the model weights to zero
    - C. It deletes the loss function
    - D. It disables backpropagation

11. Which metric is most useful when false negatives are very costly?
    - A. Recall
    - B. Training loss only
    - C. Number of model parameters
    - D. Number of convolution filters only

12. In transfer learning with a pretrained CNN, a common first approach is:
    - A. discard all pretrained weights
    - B. freeze early feature layers and train a new classifier head
    - C. train only on labels from ImageNet
    - D. remove all convolution layers

13. Batch normalization uses different statistics in training and evaluation. What should be done before validation?
    - A. `model.eval()`
    - B. `loss.backward()`
    - C. `optimizer.step()`
    - D. `model.zero_parameters()`

14. In `df.groupby("customer_id").agg(unique_products=("product_id", "nunique"))`, what does `nunique` compute?
    - A. the number of distinct non-null products per customer
    - B. the total number of rows per customer
    - C. the most frequent product per customer
    - D. the average product price per customer

15. What does padding in convolution mainly control?
    - A. number of classes
    - B. spatial size and edge information handling
    - C. optimizer type
    - D. random seed

16. If training loss decreases but validation loss increases for many epochs, the likely issue is:
    - A. underfitting
    - B. overfitting
    - C. missing `import torch`
    - D. too few labels in the output layer only

17. What is the main benefit of using `DataLoader` with mini-batches?
    - A. It converts classification to regression
    - B. It handles batching, shuffling, and efficient iteration
    - C. It removes the need for labels
    - D. It guarantees perfect generalization

18. Which statement about logits is correct?
    - A. Logits are raw unnormalized scores before softmax
    - B. Logits must sum to 1
    - C. Logits are always binary
    - D. Logits are labels after encoding

19. In image classification, global average pooling can help by:
    - A. increasing spatial dimensions
    - B. reducing parameters compared with a large flatten + linear head
    - C. removing all channels
    - D. preventing gradient computation

20. In pandas, `groupby().agg()` is preferred over Python loops mainly because:
    - A. it is usually more concise and faster for tabular aggregation
    - B. it prevents all data leakage automatically
    - C. it trains neural networks
    - D. it always uses GPU acceleration

## Section B: Conceptual Questions

Answer any 4 questions. Each question carries 5 marks.

1. You are given product interaction logs with `customer_id`, `event_type`, `product_id`, `price`, `quantity`, and `is_returned`. Describe a pandas feature engineering pipeline that creates one row per customer using counts, revenue, return behavior, unique products, and conversion-rate style features.

2. A CNN trained on 28x28 grayscale images reaches 99% training accuracy but only 82% validation accuracy. Explain at least four likely causes or fixes.

3. Explain the difference between logits, probabilities, and predicted class labels in a multi-class PyTorch classifier. Include where `CrossEntropyLoss` fits in.

4. A medical image dataset has 5,000 normal samples and 300 disease-positive samples. Propose a training and evaluation strategy that handles imbalance responsibly.

5. You are fine-tuning a pretrained CNN on a small custom dataset. Explain when you would freeze layers, when you would unfreeze layers, and how you would choose learning rates.

## Section C: Coding

Complete `coding/student.py`. tests are in `coding/tests/test_student.py`.
Use `coding/sample_io.md` for fixed examples of input and output.

### Task 1: Pandas Feature Engineering

Implement:

```python
build_customer_product_features(events)
```

Requirements:

- Build one row per `customer_id`.
- Create these columns:
  - `customer_id`
  - `view_count`
  - `cart_count`
  - `purchase_count`
  - `non_returned_purchase_count`
  - `gross_revenue`
  - `return_rate`
  - `unique_products`
  - `avg_order_value`
  - `cart_to_purchase_rate`
  - `view_to_purchase_rate`
- Treat revenue as `price * quantity` for non-returned purchase rows.
- If a denominator is 0 for a rate feature, return 0 for that rate.

Marks: 18

### Task 2: Product Revenue Ranking

Implement:

```python
top_products_by_revenue(events, top_n=3)
```

Requirements:

- Use only non-returned purchase rows for `gross_revenue`.
- `gross_revenue = price * quantity`.
- Return these columns:
  - `product_id`
  - `non_returned_units`
  - `gross_revenue`
  - `unique_buyers`
  - `return_rate`
- Sort by `gross_revenue` descending, then `product_id` ascending.
- Return only the top `top_n` rows.
- Use 0 for `return_rate` when there are no purchases.

Marks: 12

### Task 3: Imbalance and CNN Shape Utilities

Implement:

```python
make_balanced_sampler_weights(labels)
conv2d_output_shape(input_hw, kernel_size, stride=1, padding=0, dilation=1)
```

Marks: 10

### Task 4: PyTorch CNN

Implement `TinyCnn` and `count_trainable_parameters`.

Expected architecture:

- Input: `(batch, 1, 28, 28)`
- Conv block 1: Conv2d 1 to 8, kernel 3, padding 1; BatchNorm; ReLU; MaxPool2d 2
- Conv block 2: Conv2d 8 to 16, kernel 3, padding 1; BatchNorm; ReLU; MaxPool2d 2
- Classifier: Dropout 0.2; Linear from `16 * 7 * 7` to `num_classes`
- For `TinyCnn(num_classes=5)`, `count_trainable_parameters(model)` should return `5221`.

Marks: 10

### Task 5: PyTorch Training Loop and Evaluation Metrics

Implement:

```python
train_one_epoch(model, dataloader, optimizer, criterion, device)
confusion_matrix(y_true, y_pred, num_classes)
macro_f1_from_confusion(cm)
```

`train_one_epoch` should return:

```python
{"loss": average_loss, "accuracy": average_accuracy}
```

Marks: 10
