# how to setup latex locally


## 1. Clone the Repository

```bash
git clone https://github.com/simonli357/COMP579.git
cd ~/COMP579/project/proposal
```

---

## 2. Install VS Code and LaTeX Tools

### Recommended Editor: [Visual Studio Code](https://code.visualstudio.com/)

Install **VS Code**, then install the following extensions:

### Required VS Code Extensions

| Extension Name          | Author      | Required |
|------------------------|-------------|----------|
| LaTeX Workshop          | James Yu    |  Yes     |
| LaTeX Utilities         | tecosaur    | Optional |
| LaTeX Snippets          | HaoyunQin   | Optional |

> To install: open VS Code → Extensions panel (`Ctrl+Shift+X`) → search and install.

---

## 3. Install LaTeX Distribution (Compiler)

### 🔸 Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install texlive-full
```

> This includes `pdflatex`, `biber`, and all required packages (~2–3 GB).

---

### 🔸 Windows

1. Download [**MiKTeX**](https://miktex.org/download)
2. During installation:
   - Allow **"Install missing packages on the fly"**

---

### 🔸 macOS

1. Download [**MacTeX**](https://tug.org/mactex/)
2. You may also install **TeX Live Utility** for package management (included in MacTeX).

---

## 4. How to Compile the Project

### Option A: Compile with VS Code

1. Open `main.tex` in VS Code.
2. Press `Ctrl + Alt + B` to **build PDF**.
3. PDF will appear in the preview pane.
4. If errors occur, check the **Logs panel** at the bottom.

### Option B: Compile in Terminal

```bash
pdflatex main.tex
```

> Run this inside the project folder.

---

## 5. Git Collaboration Workflow

### Pull latest changes:
```bash
git pull origin main
```

### Add and commit your edits:
```bash
git add .
git commit -m "Added section X / updated formatting"
git push origin main
```

> Always pull before starting to avoid merge conflicts.
