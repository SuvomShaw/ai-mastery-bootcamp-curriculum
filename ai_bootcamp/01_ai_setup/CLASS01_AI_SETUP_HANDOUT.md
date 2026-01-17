## Class 01 — AI Setup Handout (Mac + Windows)

Goal: by the end of this handout you can run **Python**, write your first program, and run **Jupyter notebooks** in VS Code (with a proper **virtual environment**).

---

## 0) What you need (downloads)
- **Python**: `https://www.python.org/downloads/`
- **VS Code**: `https://code.visualstudio.com/download`
- **Google Colab**: `https://colab.research.google.com/`

Tip: Only install from official links above (avoid random installers).

---

## 1) Download Python (Mac vs Windows)

### Mac (recommended)
1. Open the Python downloads page: `https://www.python.org/downloads/`
2. Download the latest **Python 3.x** for macOS.
3. Run the installer package and complete installation.

Notes for Mac:
- You will typically use `python3` (not `python`) in Terminal.
- If you see permission prompts, allow installation.

### Windows (recommended)
1. Open the Python downloads page: `https://www.python.org/downloads/`
2. Download the latest **Python 3.x** for Windows.
3. Run the installer.
4. IMPORTANT: On the first installer screen, check:
   - **“Add python.exe to PATH”**
5. Click Install.

---

## 2) Install Python and ensure it’s in PATH (Add to PATH)

### What is PATH (simple)
PATH is how your computer finds commands like `python` from any folder in the terminal.

### Windows: confirm “Add to PATH”
- If you checked **“Add python.exe to PATH”**, you’re usually done.
- If you forgot, easiest fix is to re-run the installer and enable it.

### Mac: PATH is usually fine
- On Mac, you will normally run Python as `python3`.
- If `python3` is not found, re-run the installer from `python.org`.

---

## 3) Python check (verify installation)

Open a terminal:
- **Mac**: Terminal (Applications → Utilities → Terminal)
- **Windows**: Command Prompt (or PowerShell)

Run these commands.

### Mac
```bash
python3 --version
python3 -c "print('Python OK')"
```

### Windows
Try:
```bash
py --version
py -c "print('Python OK')"
```

If `py` doesn’t work, try:
```bash
python --version
python -c "print('Python OK')"
```

---

## 4) VS Code download + install
1. Download: `https://code.visualstudio.com/download`
2. Install VS Code
3. Open VS Code

Recommended: pin VS Code to your dock/taskbar so you can open it quickly.

---

## 5) Install Python + Jupyter extensions in VS Code
1. Open VS Code
2. Go to **Extensions**
   - Mac: View → Extensions
   - Windows: View → Extensions
3. Install:
   - **Python** (Microsoft): `https://marketplace.visualstudio.com/items?itemName=ms-python.python`
   - **Jupyter** (Microsoft): `https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter`
   - **Pylance** (Microsoft): `https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance`

Helpful docs:
- VS Code Python environments: `https://code.visualstudio.com/docs/python/environments`
- VS Code extensions directory: `https://marketplace.visualstudio.com/VSCode`

---

## 6) Run your first Python program (Hello World)

### Option A (fastest): run a `.py` file
1. Create a folder for classwork (example: `AI_Bootcamp`)
2. Open it in VS Code: File → Open Folder
3. Create a file: `hello.py`
4. Add:

```python
print("Hello, World!")
```

5. Open VS Code Terminal:
   - Terminal → New Terminal

Run:
- **Mac**:
```bash
python3 hello.py
```
- **Windows**:
```bash
py hello.py
```

### Option B: Python REPL (quick test)
- **Mac**: `python3`
- **Windows**: `py`

Then type:
```python
print("Hello from Python")
```

Exit:
- Mac/Linux: Ctrl+D
- Windows: Ctrl+Z then Enter

---

## 7) Recommended folder structure (for students)
Create a clean structure so your work stays organized.

Example:
```text
data_science/
  01_ai_setup/
    hello.py
    notes.md
  02_python_basics/
  03_pandas/
  04_ml/
```

Rule: one topic = one folder.

---

## 8) How to access Google Colab (backup option)
Colab runs Python in your browser (no local install needed).

1. Open: `https://colab.research.google.com/`
2. Sign in with your Google account
3. Click **New Notebook**
4. Run a cell:

```python
print("Hello from Colab")
```

When to use Colab:
- Your local Python install is failing
- You need a quick cloud notebook

Limitation:
- Colab is not ideal for practicing local file/folder automation on your laptop.

---

## 9) How to access Jupyter in VS Code

### Option A: Open an existing notebook
1. In VS Code, open a file ending with `.ipynb`
2. VS Code will open the notebook interface automatically

### Option B: Create a new notebook
1. In VS Code: File → New File
2. Choose **Jupyter Notebook**
3. Save it as: `notebook.ipynb`

### Select the correct kernel (important)
Top-right of the notebook, choose the **Python Kernel** that matches your environment.

---

## 10) How to run Jupyter cells
In a `.ipynb` notebook:
- Run a cell: **Shift + Enter**
- Run and add new cell below: **Alt/Option + Enter** (varies by OS)
- Restart kernel (fixes many issues): use the notebook toolbar → **Restart**

Notebook rule:
- If you run cells out of order, variables may not exist.

---

## 11) Create a virtual environment (venv) in VS Code (what + why + how)

### What is a venv (simple)
A **virtual environment** is an isolated Python environment for one project.

### Why we use venv
- Keeps each project’s libraries separate
- Prevents “it works on my machine” problems
- Makes installs predictable

### Create and activate venv (recommended commands)
Open VS Code Terminal inside your project folder.

#### Mac
Create:
```bash
python3 -m venv ai_env
```
Activate:
```bash
source ai_env/bin/activate
```

#### Windows (Command Prompt)
Create:
```bash
py -m venv ai_env
```
Activate:
```bash
ai_env\Scripts\activate
```

Success sign: your terminal shows something like `(ai_env)` at the start.

### Make VS Code use your venv (important)
1. Open Command Palette:
   - Mac: Cmd + Shift + P
   - Windows: Ctrl + Shift + P
2. Search: **Python: Select Interpreter**
3. Choose the interpreter inside `ai_env`


Then in your notebook’s kernel selector choose **Python (ai_env)**.

Official venv docs: `https://docs.python.org/3/library/venv.html`

---

## Quick troubleshooting (common)
- **“python not found”**
  - Mac: try `python3 --version`
  - Windows: try `py --version`
- **Installed packages but imports fail**
  - You are likely using the wrong interpreter/kernel. Re-select the venv in VS Code and notebook kernel.
- **Notebook kernel missing**
  - Install `ipykernel` and register the kernel (see Step 11).

