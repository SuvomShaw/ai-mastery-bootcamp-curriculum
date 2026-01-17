## Class 01 — Basics + History of AI (Student Notes)

These notes are meant to be **simple, readable, and memorable**.

---

## 1) What is Data Science?
**Data Science** is the practice of using data to answer questions and support decisions.

Typical work includes:
- Collecting data (files, databases, APIs)
- Cleaning data (fix missing values, duplicates)
- Exploring data (charts, summaries)
- Modeling (statistics / ML)
- Communicating results (insights, dashboards, reports)

**Example:** “Will customers cancel their subscription next month?” (predict churn)

---

## 2) What is AI vs ML?

### AI (Artificial Intelligence)
**AI** is the broad goal: making machines do tasks that normally require human intelligence (reasoning, language, planning, perception).

### ML (Machine Learning)
**ML** is a way to build AI systems by learning patterns from data (instead of writing every rule by hand).

Simple relationship:
- **AI** is the big umbrella
- **ML** is one approach inside AI

---

## 3) How systems evolved (very high level)

### Diagram (easy to remember)
```text
Rules (hand-coded)  ->  Statistics (math models)  ->  ML (learn from data)
                                      |
                                      v
                           Deep Learning (neural nets, big data, GPUs)
                                      |
                                      v
                      Generative AI (create text/images/code)
                                      |
                                      v
                     LLM Agents (LLM + tools + planning + actions)
```

### Timeline idea (not exact dates)
- Early: **Rule-based / expert systems**
- Later: **Statistics + probabilistic modeling**
- 2000s+: **Machine learning** becomes mainstream
- 2010s+: **Deep learning** breakthroughs (vision, speech)
- 2020s+: **Generative AI** + large language models
- Now: **Agents** that can use tools and do multi-step tasks

---

## 4) What is a rule-based system?
A **rule-based system** uses human-written rules like:

```text
IF condition THEN action
```

**Example (spam filtering, simplified):**
- IF email contains “win money” AND has suspicious link → mark as spam

**Pros:**
- Easy to understand and debug

**Cons:**
- Breaks when rules don’t cover real-world variety
- Hard to maintain (too many rules over time)

---

## 5) What is Statistics?
**Statistics** is the field of learning from data using mathematics.

You use statistics to:
- Summarize data (mean, median, variance)
- Estimate unknown values (confidence intervals)
- Test hypotheses (A/B tests)
- Build simple predictive models (regression)

---

## 6) Quantitative vs Qualitative data

### Quantitative (numbers you can measure)
- Price, age, height, number of orders, time spent on app
- Easy to compute averages, trends, correlations

### Qualitative (categories / text / meaning)
- Product category, user feedback text, sentiment, interview notes
- Often needs labeling or text processing to analyze at scale

Note: you can convert qualitative → quantitative (example: “positive/negative” sentiment as 1/0).

---

## 7) What is statistical modeling? (with a simple example)
**Statistical modeling** means using math equations to describe relationships in data.

### Example: Linear Regression (very simple)
Goal: predict house price using size.

```text
price ≈ (a * size) + b
```

Where:
- `a` and `b` are learned from past data

Use case:
- If size increases, price usually increases (but not perfectly).

---

## 8) What is Machine Learning? (with an example)
**Machine Learning** learns patterns from data to make predictions/decisions.

### Example: Predict if a student will pass
Input features:
- hours studied, attendance, previous scores
Output:
- pass / fail (classification)

In ML:
- You give examples (training data)
- The model learns a decision rule from data

---

## 9) What is Deep Learning? (with an example)
**Deep Learning** is a type of ML using **neural networks** with many layers.

### Example: Image recognition
Input:
- image pixels
Output:
- “cat” or “dog”

Deep learning works well when:
- You have large datasets
- The patterns are complex (images, audio, language)

---

## 10) What is Generative AI? (with an example)
**Generative AI** creates new content (text, images, code, audio).

### Example: Generate a product description
Input:
- bullet points about a product
Output:
- a readable marketing paragraph

It doesn’t just classify; it **generates**.

---

## 11) What are LLMs and Agents? (simple)

### LLM (Large Language Model)
An **LLM** is trained on lots of text and learns to predict the next token (word/part of a word).

**Example:**
- Summarize a PDF
- Explain code
- Draft an email

### Agent (LLM Agent)
An **agent** is typically:
- an LLM
- plus **tools** (browser/search, calculator, code execution, database, APIs)
- plus a loop to do **multi-step** work (plan → act → check → act)

**Example agent task (simple):**
- “Find 3 laptop options under $800, compare specs, and output a short recommendation.”
  - The agent searches, extracts info, compares, writes a summary.

Important note:
- Agents can be powerful, but they can also make mistakes. Always verify important outputs.

