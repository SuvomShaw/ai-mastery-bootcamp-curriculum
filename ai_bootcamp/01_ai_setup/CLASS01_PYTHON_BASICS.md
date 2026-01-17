## Class 01 — Python Basics (Student Notes + Practice)

Goal: understand the **core Python building blocks** you’ll use in AI/data science and practice with small, runnable examples.
Key takeaway: **Python is the most common language for AI and data science because it is easy to use and has powerful libraries.**

---

## 1) What is Python?
**Python** is a programming language that is:
- Easy to read and write
- Great for automation and data work
- Supported by a huge ecosystem of libraries

---

## 2) Why Python in AI / Data Science?
Python is popular in AI because:
- It has strong libraries: **NumPy**, **Pandas**, **Matplotlib/Seaborn**, **scikit-learn**, **PyTorch**, **TensorFlow**
- Most AI tutorials and tools assume Python
- It is fast to prototype, experiment, and iterate

Typical AI workflow in Python:
```text
Load data -> Clean data -> Explore -> Train model -> Evaluate -> Improve -> Deploy
```

---

## 3) Running Python (quick reminder)

### Run a script
- Mac:
```bash
python3 hello.py
```
- Windows:
```bash
py hello.py
```

### Run in a notebook
Use a `.ipynb` file in VS Code and run cells with **Shift + Enter**.

---

## 4) Variables (names for values)
```python
name = "Asha"
age = 22
is_student = True
```

Check the type:
```python
type(name), type(age), type(is_student)
```

---

## 5) Core data types (with examples)

### 5.1 Numbers: `int`, `float`
```python
x = 10        # int
y = 3.14      # float
total = x + y
print(total)
```

### 5.2 Boolean: `bool`
```python
is_adult = age >= 18
print(is_adult)  # True/False
```

### 5.3 Strings: `str`
```python
text = "  Hello AI  "
print(text.strip().lower())  # "hello ai"
```

Useful string methods:
- `lower()`, `upper()`, `strip()`, `replace()`, `split()`

f-strings (recommended formatting):
```python
city = "Bangalore"
print(f"My city is {city}")
```

---

## 6) Collections (most-used in data work)

### 6.1 List (ordered, mutable)
Use lists when you need an ordered collection you may change.
```python
scores = [10, 20, 30]
scores.append(40)
print(scores)  # [10, 20, 30, 40]
```

Indexing and slicing:
```python
nums = [5, 6, 7, 8, 9]
print(nums[0])     # first item
print(nums[-1])    # last item
print(nums[1:4])   # items at index 1,2,3
```

### 6.2 Tuple (ordered, immutable)
Use tuples when values should not change.
```python
point = (10, 20)
print(point[0], point[1])
```

### 6.3 Dict (key-value mapping)
Use dictionaries to store labeled data.
```python
student = {"name": "Asha", "age": 22, "city": "Bangalore"}
print(student["name"])
student["age"] = 23
print(student)
```

Common dict patterns:
```python
for key, value in student.items():
    print(key, value)
```

### 6.4 Set (unique values)
Use sets to remove duplicates.
```python
ids = [101, 101, 102, 103, 103]
unique_ids = set(ids)
print(unique_ids)  # {101, 102, 103}
```

---

## 7) Control flow (if / for / while)

### `if` condition
```python
score = 72
if score >= 60:
    print("Pass")
else:
    print("Fail")
```

### `for` loop
```python
names = ["Asha", "Ravi", "Meera"]
for n in names:
    print(n)
```

`enumerate()` gives index + value:
```python
for i, n in enumerate(names, start=1):
    print(i, n)
```

---

## 8) Functions (reusable logic)
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

print(greet("Asha"))
```

Why functions matter:
- Less copy-paste
- Easier testing
- Cleaner notebooks/scripts

---

## 9) Practice (run these yourself)
Create a file `practice_basics.py` (or a notebook) and solve:

### Practice A — Strings
1. Take: `s = "  Python for AI  "`
2. Print:
   - stripped version
   - lowercase version
   - word list using `split()`

### Practice B — Lists
1. Create a list of 5 numbers
2. Print:
   - sum of the numbers
   - max number
   - a new list with each number multiplied by 2

### Practice C — Dict
1. Create a dict for a book: title, author, year
2. Print a sentence using an f-string:
   - `"<title>" by <author> (<year>)`

### Practice D — Simple ML-style thinking
You have marks:
```python
marks = [55, 72, 91, 38, 66, 72]
```
1. Count how many students passed (>= 60)
2. Count how many got distinction (>= 85)
3. Find unique marks (use a `set`)

---

## 10) Next steps (AI-ready Python)
After these basics, you’ll use:
- **Pandas** for tabular data
- **NumPy** for arrays and math
- **Matplotlib/Seaborn** for plots
- **scikit-learn** for ML models

