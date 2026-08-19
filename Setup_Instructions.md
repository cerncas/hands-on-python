# Hands-On Lattice (and Longitudinal) Calculations using Python — Setup Instructions

---

During the course we will use **Python 3** in a **Jupyter notebook** with [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) and, mostly, the [numpy](https://numpy.org/) and [matplotlib](https://matplotlib.org/) packages. A basic knowledge of Python is assumed; if you are new to it, see the [very short introduction](#a-very-short-introduction-to-python) below. The rest of the document explains how to install everything **on your own laptop**, step by step, for **Linux**, **macOS** and **Windows**.

> **Important:** please go through this document **before coming to CAS**, so that both **you** and **your laptop** are ready. It takes 20–40 minutes with a good internet connection.

> **In a hurry?** The [Quick Start](./Setup_QuickStart.md) is the same setup in one page, without explanations. If something does not work, come back here: the [Troubleshooting](#troubleshooting) section lists the problems we see most often. If you are still stuck, find a tutor at the school — but please try beforehand.

> **There is a treasure hidden at the end of this document.** Find it, and bring it to your 1-slide, 1-minute introduction on Monday.

---

## Table of contents

- [A very short introduction to Python](#a-very-short-introduction-to-python)
  - [Test Python on a web page](#test-python-on-a-web-page)
- [Software setup](#software-setup)
  - [What we will install](#what-we-will-install)
  - [Which route should I take?](#which-route-should-i-take)
  - [Step 1 — Install a Python distribution](#step-1--install-a-python-distribution)
    - [Linux](#linux)
    - [macOS](#macos)
    - [Windows](#windows)
  - [Step 2 — Create the course environment](#step-2--create-the-course-environment)
  - [Step 3 — Download the course material](#step-3--download-the-course-material)
  - [Step 4 — Launch JupyterLab](#step-4--launch-jupyterlab)
  - [Other ways to run the notebooks](#other-ways-to-run-the-notebooks)
- [Test that everything works!](#test-that-everything-works)
  - [Check that you are in the right environment](#check-that-you-are-in-the-right-environment) · [Imports](#imports) · [Indexing](#indexing) · [Implicit loops](#implicit-loops) · [Linear algebra](#linear-algebra) · [Plotting](#plotting) · [Pandas](#using-pandas-dataframes) · [Widgets](#animations-and-widgets-optional)
  - [The one-cell self-check](#the-one-cell-self-check)
  - [The treasure hunt](#the-treasure-hunt)
- [Troubleshooting](#troubleshooting)
- [Appendix A: Python packages and cheatsheets](#appendix-a-python-packages-and-cheatsheets)
- [Appendix B: A minimal terminal survival kit](#appendix-b-a-minimal-terminal-survival-kit)
- [Appendix C: A minimal conda survival kit](#appendix-c-a-minimal-conda-survival-kit)

---

# A very short introduction to Python

There are many good courses and videos on the internet. Two suggestions from YouTube:

[![Python for Beginners - Learn Python in 1 Hour](http://img.youtube.com/vi/kqtD5dpn9C8/0.jpg)](http://www.youtube.com/watch?v=kqtD5dpn9C8)
[![Learn Python - Full Course for Beginners](http://img.youtube.com/vi/rfscVS0vtbw/0.jpg)](http://www.youtube.com/watch?v=rfscVS0vtbw)

Two more references worth bookmarking:

- the [official Python tutorial](https://docs.python.org/3/tutorial/);
- the [Scientific Python lectures](https://lectures.scientific-python.org/), a free introduction to `numpy`, `matplotlib` and `scipy` — exactly the part of Python we will use.

You do not need much: variables, lists, `for` loops, `if` statements, functions, and how to call a method on an object. We will introduce the rest as we go.

### Test Python on a web page

If you do not have Python installed yet, you can try simple snippets on the web without installing anything. Connect, for example, to

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cerncas/hands-on-python.git/HEAD)

and test the following commands:

```python
import numpy as np

# Matrix definition
Omega = np.array([[0, 1], [-1, 0]])
M = np.array([[1, 0], [1, 1]])

# Sum and multiplication of matrices
Omega - M.T @ Omega @ M
# M.T means the "transpose of M".

# Function definition
def Q(f=1):
    return np.array([[1, 0], [-1/f, 1]])

# Eigenvalues and eigenvectors
np.linalg.eig(M)
```

You can compare your output with the ones [here](tests/SimpleTest.ipynb).

> **Note:** Binder is a free, shared service: it can be slow to start, and **your work is lost when the session ends** unless you download the notebooks. Fine as a first contact with Python and as an emergency fallback, but **not** a replacement for a local installation.

---

# Software setup

## What we will install

Four things, which are often confused with each other:

| # | What | Why | What we use |
|---|------|-----|-------------|
| 1 | A **Python interpreter** | Runs the code | Python 3.11, 3.12 or 3.13 |
| 2 | A **package manager** | Installs `numpy`, `matplotlib`, … | `conda` (recommended) or `pip` |
| 3 | An **isolated environment** | Keeps the course packages separate from the rest of your laptop | a `conda` environment called `cas` |
| 4 | An **interface** | Where you type and run the code | JupyterLab (recommended) |

The packages: `numpy`, `matplotlib`, `scipy`, `pandas`, `seaborn`, `jupyterlab`, and `ipywidgets` (for animations — nice to have, but not essential; see [below](#animations-and-widgets-optional)).

> **Why a separate environment?** It is a self-contained folder with its own Python and packages. Nothing you install for CAS can break your other projects, and if you damage it you can delete and rebuild it in two minutes.

> **Python version.** Python 3.9 and older are [end-of-life](https://devguide.python.org/versions/) and some packages no longer support them. Use **3.11, 3.12 or 3.13**. We pin **3.12** below because we tested it most. Python 3.14 is still new and some packages lag behind, so please avoid it for the course.

> **A note on Anaconda.** Earlier editions recommended the full **Anaconda Distribution**. Since March 2024, Anaconda's [Terms of Service](https://www.anaconda.com/legal/terms/terms-of-service) require a paid licence for organisations with 200 or more employees, including government and non-profit entities. Universities stay exempt for course teaching (see Anaconda's [statement for academia](https://www.anaconda.com/blog/update-on-anacondas-terms-of-service-for-academia-and-research)), but the exemption is unclear for research institutes and national laboratories — where many CAS participants work.
>
> We therefore recommend **Miniforge**: the same `conda` tool, but taking packages only from the free [conda-forge](https://conda-forge.org/) channel. It is also much smaller (~1 GB instead of ~5 GB). **If you already have Anaconda and it works, keep it** — just see the [note on channels](#i-get-a-terms-of-service-prompt-from-conda) if conda asks you to accept its terms.

## Which route should I take?

Pick **one** row. All routes end with a working `cas` environment.

| Your situation | Route | Section |
|---|---|---|
| No Python, or not sure what I have | **Miniforge** | [Linux](#linux) · [macOS](#macos) · [Windows](#windows) |
| I have Anaconda / Miniconda and it works | Keep it | [Step 2](#step-2--create-the-course-environment) |
| Linux, and I prefer the system Python | `venv` + `pip` | [Option L2](#option-l2--system-python--venv) |
| Windows, but I would rather work in Linux | **WSL** | [Option W3](#option-w3--wsl-linux-inside-windows) |
| I would rather not use a terminal | Graphical installer + JupyterLab Desktop | [macOS](#macos) / [Windows](#windows), then [here](#other-ways-to-run-the-notebooks) |
| Nothing works on my laptop | Binder, as a fallback | [Binder](#test-python-on-a-web-page) |

**Disk space:** about 1 GB for Miniforge and the `cas` environment (~5 GB for the full Anaconda Distribution).

---

## Step 1 — Install a Python distribution

> Skip this step if you already have a working `conda`. Type `conda --version` in a terminal: if it answers with a version number, go to [Step 2](#step-2--create-the-course-environment).

### Linux

#### Option L1 — Miniforge (recommended)

Open a terminal (`Ctrl`+`Alt`+`T` on most distributions) and run:

```bash
# 1. Download the installer for your machine
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# 2. Run it
bash "Miniforge3-$(uname)-$(uname -m).sh"
```

The shell fills in `$(uname)` and `$(uname -m)` for you, so these two lines work on both Intel and ARM machines. During the installation:

- press `Enter` to read the licence, then type `yes` to accept it;
- press `Enter` to accept the default location (`~/miniforge3`);
- answer **`yes`** when asked whether to initialise conda in your shell profile.

Then **close the terminal and open a new one** — the initialisation only applies to new shells. Your prompt should now start with `(base)`. Check it:

```bash
conda --version     # e.g. conda 26.3.2
```

Optional, but recommended — stop conda from activating `base` in every terminal:

```bash
conda config --set auto_activate_base false
```

If `curl` is missing, use `wget` with the same URL, or install curl (`sudo apt install curl`, or `sudo dnf install curl` on Fedora).

> **No admin rights?** None of this needs `sudo`: Miniforge installs inside your home directory.

Now go to [Step 2](#step-2--create-the-course-environment).

#### Option L2 — System Python + venv

If you prefer the Python of your distribution, that is fine — but you **must** work inside a virtual environment.

```bash
# 1. Check the version: we need 3.11 or newer
python3 --version

# 2. Make sure venv and pip are available (Debian/Ubuntu; adapt for your distro)
sudo apt install python3-venv python3-pip

# 3. Create and activate the environment
python3 -m venv ~/cas-venv
source ~/cas-venv/bin/activate

# 4. Install the packages
python -m pip install --upgrade pip
python -m pip install numpy matplotlib seaborn scipy ipywidgets jupyterlab pandas
```

Your prompt should now start with `(cas-venv)`. **Run `source ~/cas-venv/bin/activate` in every new terminal** before working on the course.

> **A common trap:** `pip install` *without* an active virtual environment gives `error: externally-managed-environment` on recent Debian, Ubuntu and Fedora. This is not a bug: your distribution is protecting its own Python. The virtual environment above is the fix. Do **not** use `--break-system-packages`.

With this route, `~/cas-venv` replaces the `cas` environment mentioned later, and you can skip [Step 2](#step-2--create-the-course-environment).

#### Option L3 — Anaconda Distribution

Download from [anaconda.com/download](https://www.anaconda.com/download) and follow the [official Linux instructions](https://docs.anaconda.com/anaconda/install/linux/). It is a ~5 GB install containing everything we need, plus the graphical Anaconda Navigator. Please read the [note on Anaconda](#what-we-will-install) first.

---

### macOS

Open the Terminal from *Launchpad → Other → Terminal*, or press `Cmd`+`Space` and type "Terminal".

#### Option M1 — Miniforge from the terminal (recommended)

```bash
# 1. Download the installer for your machine
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# 2. Run it
bash "Miniforge3-$(uname)-$(uname -m).sh"
```

Accept the licence, accept the default path (`~/miniforge3`), and answer **`yes`** when asked to initialise your shell. Then **quit Terminal and open it again**. Your prompt should start with `(base)`.

```bash
conda --version
conda config --set auto_activate_base false   # optional but recommended
```

#### Option M2 — Miniforge graphical installer

Signed and notarised `.pkg` installers are available. First check which chip you have with `uname -m`: `arm64` means Apple Silicon (M1/M2/M3/M4…), `x86_64` means Intel. Then download the matching file from the [Miniforge releases page](https://github.com/conda-forge/miniforge/releases/latest) (`Miniforge3-MacOSX-arm64.pkg` or `Miniforge3-MacOSX-x86_64.pkg`) and double-click it.

You can change the location with *"Change Install Location"*. Importantly, make sure the option to **initialise conda for your shell** is ticked — otherwise `conda` will not be found in the terminal.

#### Option M3 — Anaconda Distribution

Download from [anaconda.com/download](https://www.anaconda.com/download) and follow the [official macOS instructions](https://docs.anaconda.com/anaconda/install/mac-os/). Please read the [note on Anaconda](#what-we-will-install) first.

> **Do not install Miniforge with Homebrew.** The Homebrew repackaging can cause incompatibilities that the Miniforge developers do not test for and do not recommend. Use the official installer.

> **macOS uses `zsh`** by default, and the installer writes to `~/.zshrc`. If you use `bash`, run `conda init bash` once after installing.

Now go to [Step 2](#step-2--create-the-course-environment).

---

### Windows

#### Option W1 — Miniforge (recommended)

1. Download **`Miniforge3-Windows-x86_64.exe`** from the [Miniforge releases page](https://github.com/conda-forge/miniforge/releases/latest).
2. Run it and:
   - choose **"Just Me"** — no administrator rights needed, and fewer permission problems later;
   - accept the default folder (`C:\Users\<yourname>\miniforge3`), **unless** your Windows user name contains spaces or accents (é, ü, ñ…). In that case use a simple path such as `C:\miniforge3`. This detail causes many strange failures;
   - leave *"Add Miniforge3 to my PATH environment variable"* **unticked** (the default).
3. Open **"Miniforge Prompt"** from the Start menu and check:

   ```bat
   conda --version
   ```

> **Use the "Miniforge Prompt" for everything in this course**, not a plain `cmd` window. If a command "is not recognized", you are usually in the wrong terminal.

**Optional — conda in PowerShell.** Run `conda init powershell` *once* in the Miniforge Prompt, then open a **new** PowerShell window. If it says that running scripts is disabled, run once in PowerShell and open another new window:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Option W2 — Anaconda Distribution

Download from [anaconda.com/download](https://www.anaconda.com/download) and follow the [official Windows instructions](https://docs.anaconda.com/anaconda/install/windows/).

![Anaconda distribution download page](_img_instructions/anaconda.png)

This gives you **Anaconda Navigator**, from which you can start JupyterLab and manage environments with a few clicks, without typing anything. It is the friendliest option if you dislike terminals, at the cost of a ~5 GB install. Please read the [note on Anaconda](#what-we-will-install) first.

Your terminal is then called **"Anaconda Prompt"**; use it wherever we write "Miniforge Prompt".

#### Option W3 — WSL, Linux inside Windows

Useful if you already know Linux, or if you will work on Linux clusters later.

1. Open **PowerShell as Administrator** (right-click the Start button → *Terminal (Admin)*) and run:

   ```powershell
   wsl --install
   ```

2. Reboot when asked. An Ubuntu window then asks you to choose a **username** and **password** (the password stays invisible while you type — that is normal).
3. From then on, open **"Ubuntu"** from the Start menu and follow the [Linux instructions](#option-l1--miniforge-recommended).

Two practical notes:

- **Keep your course files in the Linux home directory** (`~/`), not under `/mnt/c/...`, which is much slower.
- `jupyter lab` prints a `http://localhost:8888/lab?token=...` address. `Ctrl`+click it, or paste it into your normal Windows browser.

#### Option W4 — python.org + venv

Fine if you do not want conda at all.

1. Install Python 3.12 from [python.org/downloads](https://www.python.org/downloads/). **Tick "Add python.exe to PATH"** on the first screen — it is easy to miss.
2. In PowerShell:

   ```powershell
   py -3.12 -m venv $HOME\cas-venv
   $HOME\cas-venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install numpy matplotlib seaborn scipy ipywidgets jupyterlab pandas
   ```

   (In `cmd`, activate with `%USERPROFILE%\cas-venv\Scripts\activate.bat`.)

Re-activate the environment in every new terminal. If activation is blocked, use the `Set-ExecutionPolicy` fix [above](#option-w1--miniforge-recommended). With this route you can skip [Step 2](#step-2--create-the-course-environment).

> **Avoid the Microsoft Store version of Python** for this course: its sandboxed file access causes problems with Jupyter that are hard to diagnose.

---

## Step 2 — Create the course environment

> Skip this step if you used a `venv` ([L2](#option-l2--system-python--venv) or [W4](#option-w4--pythonorg--venv)).

The commands are **the same on Linux, macOS and Windows**. On Windows, type them in the **Miniforge Prompt**.

```bash
conda create -n cas -c conda-forge python=3.12 numpy scipy matplotlib pandas seaborn ipywidgets ipympl jupyterlab notebook
```

Answer `y` to proceed. This downloads a few hundred MB and takes a few minutes.

A more reproducible alternative is to save this as `environment.yml` and build from it:

```yaml
name: cas
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy
  - scipy
  - matplotlib
  - pandas
  - seaborn
  - ipywidgets      # interactive sliders
  - ipympl          # interactive matplotlib figures
  - jupyterlab
  - notebook
```

```bash
conda env create -f environment.yml
```

> The environment is called `cas` because of the `name:` field. If you build from the `environment.yml` shipped with `hands-on-lattice-exercises`, it will take the name written there — check with `conda env list`.

**Activate it and check what you got:**

```bash
conda activate cas
python --version          # Python 3.12.x
python -c "import numpy, matplotlib, scipy, pandas, seaborn, ipywidgets; print('all good')"
```

Your prompt should now start with `(cas)`.

> ### The golden rule
>
> **`conda activate cas` first, then everything else.** Every new terminal starts with no environment active. If you forget, you are using a different Python than you think: packages will look missing and widgets will not draw. When something is strange, ask first: *does my prompt say `(cas)`?*

To update the environment later:

```bash
conda activate cas
conda env update -f environment.yml --prune
```

---

## Step 3 — Download the course material

The exercises are in [github.com/cerncas/hands-on-lattice-exercises](https://github.com/cerncas/hands-on-lattice-exercises).

**With git** (recommended — you can then get last-minute fixes with `git pull`):

```bash
cd ~/Documents                 # on Windows: cd %USERPROFILE%\Documents
git clone https://github.com/cerncas/hands-on-lattice-exercises.git
cd hands-on-lattice-exercises
```

If `git` is missing: `conda install -c conda-forge git`, or `sudo apt install git`, or [git-scm.com](https://git-scm.com/downloads).

**Without git:** download the [ZIP archive](https://github.com/cerncas/hands-on-lattice-exercises/archive/refs/heads/master.zip) and unpack it somewhere you will find again. Avoid folder names with unusual characters.

> **If you use OneDrive, iCloud or Dropbox:** a working directory inside a syncing folder can produce file-locking problems and duplicated notebooks such as `03_Periodic_Systems (conflicted copy).ipynb`. A plain local folder is safer.

The folder containing `00_Introduction.ipynb` and `tracking_library.py` is your **working directory** for the rest of the course.

---

## Step 4 — Launch JupyterLab

1. Open a terminal (**Miniforge Prompt** on Windows).

2. **Activate the environment:**

   ```bash
   conda activate cas
   ```

3. **Go to your working directory.** This matters: JupyterLab only sees files at or below the folder where you start it.

   ```bash
   cd ~/Documents/hands-on-lattice-exercises          # Linux, macOS, WSL
   ```

   ```bat
   cd %USERPROFILE%\Documents\hands-on-lattice-exercises   :: Windows
   ```

   If you are lost, see [Appendix B](#appendix-b-a-minimal-terminal-survival-kit).

4. **Launch:**

   ```bash
   jupyter lab
   ```

5. Your browser opens on a page like this, listing the files of your working directory. Double-click `00_Introduction.ipynb`, or create a new "Python 3" notebook from the launcher.

   ![The JupyterLab interface](_img_instructions/upload_5b0618b75e4f4df0facf2a609b9354b5.png)

6. **To stop JupyterLab**, press `Ctrl`+`C` twice in the terminal, or use *File → Shut Down*. Closing the browser tab does **not** stop it.

> If no browser opens, copy the `http://localhost:8888/lab?token=...` line from the terminal into your browser. Keep the `?token=...` part: it is what identifies you.

If you are new to Python, this [example notebook](tests/PythonExample.ipynb) (courtesy of *Simon Albright*) is a gentle start. Then go to [Test that everything works!](#test-that-everything-works) — please do this before the school.

---

## Other ways to run the notebooks

JupyterLab in the browser is what the tutors will use. You are free to use something else:

| Tool | Good for | Watch out for |
|---|---|---|
| **`jupyter notebook`** | The classic, simpler interface. Run it like `jupyter lab`. | Nothing special; Notebook 7 is built on JupyterLab. |
| **[JupyterLab Desktop](https://github.com/jupyterlab/jupyterlab-desktop)** | The same JupyterLab as an application — no terminal, no browser tab. | Point it to your `cas` environment in *Settings → Python environment*, otherwise it uses its own Python and your packages seem to disappear. |
| **[Visual Studio Code](https://code.visualstudio.com)** | Good notebook support, plus a real editor and debugger. Install the *Python* and *Jupyter* extensions. | Select the kernel explicitly: click *Select Kernel* at the top right and choose `cas`. |
| **[Anaconda Navigator](https://docs.anaconda.com/navigator/)** | Fully graphical: choose the `cas` environment, then click "Launch" under JupyterLab. | Only with the full Anaconda Distribution. |
| **[Spyder](https://www.spyder-ide.org/)** | A MATLAB-like IDE, pleasant for `.py` scripts, with a variable explorer. | **Not ideal here.** Notebooks need the separate [`spyder-notebook`](https://github.com/spyder-ide/spyder-notebook) plugin, which does not work with the standalone Spyder installers. If you want it: `conda install -c conda-forge spyder spyder-notebook`. |
| **[Binder](https://mybinder.org/v2/gh/cerncas/hands-on-lattice-exercises/HEAD)** | No installation, works in any browser. | **Your work is lost when the session ends** unless you download the notebooks. Fallback only. |

Whichever you choose, the [tests below](#test-that-everything-works) must pass inside it.

---

# Test that everything works!

> Please run **all** the examples below: they check your installation and introduce the Python style we will use. Work in a new notebook in your working directory, or download [`test_notebook.ipynb`](./tests/test_notebook.ipynb) from this repository. Run each block in its own cell, with `Shift`+`Enter`.

### Check that you are in the right environment

This single cell answers most of the questions we get:

```python
import sys, platform
print("Python version:", platform.python_version())
print("Interpreter:   ", sys.executable)
```

The path should contain `cas` (or `cas-venv`), for example `/home/yourname/miniforge3/envs/cas/bin/python`. If it points to `base`, to `/usr/bin/python3`, or to something you do not recognise, your notebook runs on the wrong Python — see [the wrong kernel](#my-notebook-runs-the-wrong-python).

### Imports

This should run without any error and print nothing:

```python
# numpy: our main numerical package
import numpy as np

# matplotlib and seaborn: our plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

# widget for producing animations
from ipywidgets import interactive

# linear algebra and optimisation algorithms
from numpy.linalg import norm
from scipy.optimize import minimize

# some other useful packages
from copy import deepcopy
import pandas as pd
```

If you get `ModuleNotFoundError`, install the missing package **in the activated environment**:

```bash
conda activate cas
conda install -c conda-forge <name-of-the-package>
```

then **restart the kernel** (*Kernel → Restart Kernel*): a running kernel does not see packages installed afterwards.

### Indexing

Generate a random array and select specific elements:

```python
import numpy as np

# Create an array
array1d = np.random.uniform(size=10)

# Print selected elements
print("Entire array: " + str(array1d) + "\n")
print("Specific element: " + str(array1d[5]) + "\n")
print("Last element: " + str(array1d[-1]) + "\n")
print("Specific elements: " + str(array1d[3:7]) + "\n")
print("First 5 elements array: " + str(array1d[:5]) + "\n")
print("Last 5 elements array: " + str(array1d[5:]) + "\n")
```

will result in, e.g.:

```text
Entire array: [0.09402447 0.05647033 0.79670378 0.60573004 0.81588777 0.97863634 0.51376609 0.19763518 0.7649532  0.59285346]

Specific element: 0.9786363385079204

Last element: 0.5928534616865488

Specific elements: [0.60573004 0.81588777 0.97863634 0.51376609]

First 5 elements array: [0.09402447 0.05647033 0.79670378 0.60573004 0.81588777]

Last 5 elements array: [0.97863634 0.51376609 0.19763518 0.7649532  0.59285346]
```

Your numbers will be different — they are random. What matters is the pattern: indices start at `0`, `-1` is the last element, and the slice `a:b` includes `a` but excludes `b`.

### Implicit loops

In contrast to programming languages like C++, Python with numpy handles whole vectors at once. No loop is needed to multiply each element by a constant, or to square it:

```python
import numpy as np

# Create an array
array1d = np.random.uniform(size=10)

print("Entire array: " + str(array1d) + "\n")
print("Each element multiplied by 5: " + str(5 * array1d) + "\n")
print("Each element squared: " + str(array1d**2) + "\n")
print("Square root of each element: " + str(np.sqrt(array1d)) + "\n")
```

will result in, e.g.:

```text
Entire array: [0.2240143  0.35153156 0.68864907 0.14062298 0.77280195 0.26872206 0.9135403  0.8776261 0.26158576 0.93883652]

Each element multiplied by 5: [1.12007151 1.75765782 3.44324537 0.70311488 3.86400975 1.34361029 4.56770149 4.3881305 1.30792879 4.69418259]

Each element squared: [0.05018241 0.12357444 0.47423755 0.01977482 0.59722285 0.07221154 0.83455588 0.77022757 0.06842711 0.88141401]

Square root of each element: [0.47330149 0.59290097 0.82984883 0.3749973  0.87909155 0.51838408 0.95579302 0.936817 0.51145455 0.96893577]
```

This "vectorised" style is shorter and much faster than an explicit `for` loop, and it is how we will write nearly all our tracking code.

### Linear algebra

```python
import numpy as np

# Matrix definition
Omega = np.array([[0, 1], [-1, 0]])
M = np.array([[1, 0], [1, 1]])

# Sum and multiplication of matrices
print(Omega - M.T @ Omega @ M)
# M.T means the "transpose of M", and @ is the matrix product.

# Function definition
def Q(f=1):
    return np.array([[1, 0], [-1/f, 1]])

# Eigenvalues and eigenvectors
print(np.linalg.eig(M))
```

gives

```text
[[0 0]
 [0 0]]
EigResult(eigenvalues=array([1., 1.]), eigenvectors=array([[ 0.00000000e+00,  2.22044605e-16],
       [ 1.00000000e+00, -1.00000000e+00]]))
```

Two remarks, both useful later:

- The first result is a matrix of zeros. This is not a coincidence: `Omega - M.T @ Omega @ M == 0` says that `M` is **symplectic**, a property every transfer matrix we build must satisfy. Remember this check — it is the cheapest way to catch a mistake in a lattice.
- With **numpy 2.x**, `np.linalg.eig` returns a named tuple shown as `EigResult(eigenvalues=..., eigenvectors=...)`. Older material shows a plain tuple. Nothing changed in substance, and `w, v = np.linalg.eig(M)` still works.

### Plotting

A simple plot:

```python
# Import numpy and plotting
import numpy as np
from matplotlib import pyplot as plt

plt.plot([0, 10], [0, 10], 'ob-')
plt.xlabel('My x-label [arb. units]')
plt.ylabel('My y-label [arb. units]')
plt.title('My title')
plt.show()
```

![A simple straight-line plot with axis labels and a title](_img_instructions/upload_b7610a37c41729a79bc4b0a5d863594b.png)

Or something fancier:

```python
# import numpy and seaborn
import numpy as np
import seaborn as sns

sns.set_theme(style="ticks")
rs = np.random.RandomState(11)
x = rs.normal(size=1000)
y = rs.normal(size=1000)
sns.jointplot(x=x, y=y, kind="hex")
```

![A hexbin joint plot of two normally distributed variables](_img_instructions/upload_1c7eaa74ee5422c62408cc9a57f7f0de.png)

> **No figure?** In a notebook, plots appear automatically. If you only get a line such as `<Figure size 640x480 with 1 Axes>`, add `%matplotlib inline` at the top of the cell. In a `.py` script you need `plt.show()`.

### Using pandas dataframes

```python
# Import pandas
import pandas as pd

# Import also seaborn, only to profit from its datasets
import seaborn as sns
planets = sns.load_dataset('planets')

# This dataset contains the list of known exoplanets from this catalog:
#   https://exoplanets.nasa.gov/exoplanet-catalog/
#   Note: the planet mass is given in number of Jupiter masses, and orbital period is in days
# The columns of the dataset are:
print(f"Dataframe columns are: {list(planets.columns)}")

# Let's look at all exoplanets which have:
#  - mass smaller than a factor 2 of Earth (mind, Jupiter is about 318 times heavier than Earth)
#  - orbital period < 400 days
myDF = planets[(planets['orbital_period'] < 400) &
               (planets['mass'] < 2/318)]
print(myDF)
```

that gives

```text
Dataframe columns are: ['method', 'number', 'orbital_period', 'mass', 'distance', 'year']
              method  number  orbital_period    mass  distance  year
46   Radial Velocity       1         3.23570  0.0036      1.35  2012
128  Radial Velocity       4         3.14942  0.0060      6.27  2005
```

> `sns.load_dataset` downloads the data the first time you call it. If you are offline or behind a strict proxy, this cell fails — harmless, since we use pandas only marginally.

### Animations and widgets (optional)

> We do use small **animations** during the course — sliders that let you change a parameter and watch the result move. They make some ideas much easier to see, so please try the cell below.
>
> But they are **optional**: every exercise can also be done without them, and no result of the course depends on a working slider. If this cell does not work, spend a few minutes on [widgets do not display](#widgets-do-not-display), and if it resists, **come to the school anyway** — a tutor will sort it out in two minutes, or you will simply follow on the screen next to you.

```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive

t = np.linspace(0, 10, 1000)

def plotIt(f):
    plt.plot(t, np.sin(2*np.pi*f*t))
    plt.grid(True)

interactive_plot = interactive(plotIt, f=(0.1, 1, .1), continuous_update=True)
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot
```

![An interactive sine plot with a slider controlling the frequency](_img_instructions/upload_0dff4499bd5e7b21942e1990cd76d0e9.png)

You should see a **slider** that redraws the sine wave when you drag it. If you instead see text such as `interactive(children=(FloatSlider(value=0.1, ...` or a box saying *"Error displaying widget"*, see [widgets do not display](#widgets-do-not-display). This is the most common installation problem, and it almost always has the same cause — but remember that it is not a blocking one.

### The one-cell self-check

Finally, run this. If every line prints a version and you get the last message, you are ready.

```python
import sys, platform
import numpy, scipy, matplotlib, pandas, seaborn, ipywidgets, IPython
from IPython.display import display

print(f"Python      {platform.python_version()}   ({sys.executable})")
print(f"numpy       {numpy.__version__}")
print(f"scipy       {scipy.__version__}")
print(f"matplotlib  {matplotlib.__version__}")
print(f"pandas      {pandas.__version__}")
print(f"seaborn     {seaborn.__version__}")
print(f"ipywidgets  {ipywidgets.__version__}")
print(f"IPython     {IPython.__version__}")

# --- these two must pass ---
assert sys.version_info >= (3, 11), "Please use Python 3.11 or newer"

import numpy as np
M = np.array([[1., 0.], [1., 1.]])
Omega = np.array([[0., 1.], [-1., 0.]])
assert np.allclose(Omega, M.T @ Omega @ M), "Symplecticity check failed (!)"

# --- this one is optional: animations ---
if int(ipywidgets.__version__.split('.')[0]) < 8:
    print("\n(!) ipywidgets is older than version 8: animations may not work.")
from ipywidgets import IntSlider
display(IntSlider(description="drag me"))

print("\nAll good — see you at CAS!")
```

You should see the version list and the final message. You should **also** see a working slider — but if you do not, see [animations and widgets](#animations-and-widgets-optional): it is not a reason to worry.

### The treasure hunt

You made it to the end, so here is the treasure.

The cell below builds **your own particle accelerator** — a different one for every person, generated from your name — sends particles around it a few hundred times, and draws what they do.

Run it, then **put the picture on your 1-slide, 1-minute introduction on Monday**. That is our proof that your laptop is ready, and it is much more fun than a screenshot of a terminal.

```python
# ============================================================
#   CAS treasure hunt: run this cell and see what you get!
# ============================================================
import hashlib
import numpy as np
import matplotlib.pyplot as plt

MY_NAME = "Your Name Here"          #  <---  WRITE YOUR OWN NAME HERE, then run the cell

# Your own ring, built from your name
if MY_NAME == "Your Name Here":
    print("Tip: replace MY_NAME by your own name to get YOUR ring!\n")
seed  = int(hashlib.sha256(MY_NAME.strip().lower().encode()).hexdigest()[:8], 16)
TUNES = [0.206, 0.212, 0.254, 0.292, 0.316, 0.318]
CMAPS = ['plasma', 'viridis', 'cool', 'autumn', 'spring', 'turbo']
Q     = TUNES[seed % len(TUNES)]
cmap  = plt.get_cmap(CMAPS[(seed // 32) % len(CMAPS)])
mu    = 2 * np.pi * Q

# One turn of the ring: a rotation, plus a sextupole kick -
def turn(x, xp):
    xp = xp + x**2
    return np.cos(mu)*x + np.sin(mu)*xp, -np.sin(mu)*x + np.cos(mu)*xp

# How far from the centre can a particle survive?
def survives(a, turns=800):
    x, xp = np.array([a]), np.array([0.0])
    for _ in range(turns):
        x, xp = turn(x, xp)
        if not np.isfinite(x[0]) or abs(x[0]) > 10:
            return False
    return True

lo, hi = 0.0, 2.0
for _ in range(40):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if survives(mid) else (lo, mid)

# Launch 26 particles and follow them, turn after turn 
x, xp = np.linspace(0.06*lo, 0.97*lo, 26), np.zeros(26)
X, XP = [], []
for _ in range(800):
    x, xp = turn(x, xp)
    lost = ~np.isfinite(x) | (np.abs(x) > 1.6*lo)
    x, xp = np.where(lost, np.nan, x), np.where(lost, np.nan, xp)
    X.append(x.copy()); XP.append(xp.copy())
X, XP = np.array(X), np.array(XP)

# Draw your ring
fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
for j in range(26):
    ax.plot(X[:, j], XP[:, j], ',', color=cmap(0.25 + 0.75*j/25), alpha=0.9)
L = 1.08 * np.nanpercentile(np.abs(np.concatenate([X.ravel(), XP.ravel()])), 99.8)
ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
ax.set_title(f"The ring of {MY_NAME}", color='w', fontsize=15, pad=12)
ax.set_xlabel("position  $x$", color='w'); ax.set_ylabel("angle  $x'$", color='w')
ax.tick_params(colors='0.5')
for sp in ax.spines.values(): sp.set_color('0.3')
ax.text(0.98, 0.02, f"CAS-{seed:08X}", transform=ax.transAxes, ha='right',
        color='0.45', fontsize=8, family='monospace')
fig.tight_layout()
fig.savefig("my_cas_ring.png", dpi=150, facecolor=fig.get_facecolor())
plt.show()

print(f"Well done {MY_NAME}, you found the treasure!")
print(f"Your discovery code is  CAS-{seed:08X}   (tune Q = {Q})")
print("\nThe picture was saved next to your notebook as 'my_cas_ring.png'.")
print("Put it on your 1-slide for Monday, and be ready to tell us:")
print("   HOW MANY ISLANDS does your ring have?")
```

You are not supposed to understand the code yet. But this is not a decoration: those islands are called **resonance islands**, they are a real feature of real machines, and they are one reason a beam can be lost. The picture is a phase space portrait — position horizontally, angle vertically — and by the middle of the week you will be producing these yourself, for lattices you have built.

> **If nothing appears**, the problem is `numpy` or `matplotlib`, not the treasure hunt — go back to the [self-check](#the-one-cell-self-check).


---

# Troubleshooting

Find your symptom, then read the corresponding section.

| Symptom | Most likely cause |
|---|---|
| `conda: command not found` / `'conda' is not recognized` | Wrong terminal, or shell not initialised |
| `CommandNotFoundError: Your shell has not been properly configured` | Same: `conda init` not run, or shell not restarted |
| `Activate.ps1 cannot be loaded ... running scripts is disabled` | Windows PowerShell execution policy |
| `ModuleNotFoundError` for a package you just installed | Wrong environment, or kernel not restarted |
| `sys.executable` shows an unexpected path | Wrong kernel |
| `interactive(children=(FloatSlider...` printed as text | Widgets not connected to the frontend |
| `Error displaying widget: model not found` | JupyterLab and `ipywidgets` in different environments |
| "Solving environment" never finishes | Mixed channels, or too many constraints |
| A Terms of Service prompt from conda | You are reaching Anaconda's own channels |
| `error: externally-managed-environment` | `pip` outside a virtual environment on Linux |
| `SSLError` / `CERTIFICATE_VERIFY_FAILED` | Proxy or captive portal |
| `Address already in use` on port 8888 | JupyterLab already running |
| Kernel dies or restarts by itself | Out of memory, or a broken environment |

---

### `conda: command not found`

Also covers `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'`.

**Windows:** you are probably in a plain `cmd` or PowerShell window. Use **"Miniforge Prompt"** (or "Anaconda Prompt") from the Start menu. For PowerShell, run `conda init powershell` once in the Miniforge Prompt, then open a new window.

**Linux / macOS:** the shell initialisation did not take effect. **Close the terminal and open a new one** — this alone usually fixes it. If not, run the initialisation by hand and open a new terminal again:

```bash
~/miniforge3/bin/conda init
```

(use your actual install path). To check that it worked, look for a `# >>> conda initialize >>>` block in `~/.bashrc` or `~/.zshrc`. Do not use `source activate`: it is obsolete and will confuse you further.

### `Activate.ps1 cannot be loaded because running scripts is disabled on this system`

Windows blocks scripts by default. Run once in PowerShell, then open a new window:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

This affects only your own account, and `RemoteSigned` still blocks unsigned scripts downloaded from the internet.

### `ModuleNotFoundError` even though I just installed the package

Three causes, most likely first:

1. **You installed into another environment.** Check with `conda list <package>` *after* `conda activate cas`. Installing while `(base)` is active does not put anything in `cas`.
2. **You did not restart the kernel.** *Kernel → Restart Kernel* in JupyterLab.
3. **Your notebook uses a different Python than your terminal.** See the next section.

A useful trick — install into the Python the notebook is actually using:

```python
import sys
!conda install --yes --prefix {sys.prefix} -c conda-forge seaborn
```

### My notebook runs the wrong Python

Run in a cell:

```python
import sys
print(sys.executable)
```

If the path does not contain `envs/cas`, the notebook uses another kernel. Fixes, easiest first:

1. **Quit JupyterLab, run `conda activate cas`, and start `jupyter lab` again from there.** This is the recommended setup and removes a whole class of problems.
2. If you deliberately run JupyterLab from `base` and want `cas` as a *kernel choice*, register it once:

   ```bash
   conda activate cas
   python -m ipykernel install --user --name cas --display-name "Python (cas)"
   ```

   Then select *"Python (cas)"* at the top right of the notebook.

3. In **VS Code**, click *Select Kernel* at the top right and choose `cas`.

### Widgets do not display

Symptoms: a cell prints `interactive(children=(FloatSlider(value=0.1, ...` as text, or shows *"Loading widget…"* forever, or *"Error displaying widget: model not found"*.

The cause is almost always that **JupyterLab and `ipywidgets` are in different environments**: you started `jupyter lab` from `base` while the kernel is `cas`.

1. Shut JupyterLab down, then:

   ```bash
   conda activate cas
   jupyter lab
   ```

   This is usually the whole story.

2. If you must run JupyterLab from another environment, install the frontend package there too:

   ```bash
   conda install -n base -c conda-forge jupyterlab_widgets
   conda install -n cas  -c conda-forge ipywidgets
   ```

3. Check your versions — `ipywidgets` 8 with JupyterLab 4 needs no manual extension at all:

   ```bash
   conda activate cas
   conda list "ipywidgets|jupyterlab"
   ```

   If you have `ipywidgets` 7.x: `conda install -c conda-forge "ipywidgets>=8"`.

4. Reload the browser page with `Ctrl`+`Shift`+`R` (`Cmd`+`Shift`+`R` on macOS). Cached JavaScript really does cause this.

5. Many pages on the internet still recommend `jupyter nbextension enable --py widgetsnbextension` or `jupyter labextension install @jupyter-widgets/jupyterlab-manager`. **Do not run these:** they are for JupyterLab 2/3 and will break a JupyterLab 4 installation.

### "Solving environment" takes forever

Rare with modern conda, which uses the fast `libmamba` solver by default. If it happens:

- **Do not mix channels.** Using only `conda-forge` (the Miniforge default) avoids most conflicts. With Anaconda, be explicit: `conda install -c conda-forge --strict-channel-priority <package>`.
- **Do not over-constrain.** Pin only the Python version, as we do above.
- **Build a fresh environment instead of upgrading an old one.** Faster and cleaner.
- If needed: `conda install -n base conda-libmamba-solver`, then `conda config --set solver libmamba`.

### I get a Terms of Service prompt from conda

Recent conda versions ask you to accept Anaconda's Terms of Service before downloading from **Anaconda's own channels** (`defaults`, `main`, `r`, …). Packages from **conda-forge** are not concerned.

If you see this prompt, some command reached the `defaults` channel. Check with:

```bash
conda config --show channels
```

You should see `conda-forge` and nothing else. To fix it for your user:

```bash
conda config --remove-key channels
conda config --add channels conda-forge
conda config --set channel_priority strict
```

See also the [note on Anaconda](#what-we-will-install), and ask your own institute if you are unsure about your status.

### `error: externally-managed-environment`

You ran `pip install` against a Python managed by your Linux distribution. Create and activate a virtual environment first — see [Option L2](#option-l2--system-python--venv). Avoid `--break-system-packages`.

### `SSLError` / `CERTIFICATE_VERIFY_FAILED` / downloads hang

You are behind a proxy, a captive portal, or a firewall that inspects TLS.

- If a hotel or conference Wi-Fi asks you to log in on a web page, do that first.
- On an institutional network, ask IT for the proxy settings, then:

  ```bash
  conda config --set proxy_servers.http http://user:pass@proxy.example.org:8080
  conda config --set proxy_servers.https http://user:pass@proxy.example.org:8080
  ```

- **Do not** disable SSL verification (`conda config --set ssl_verify false`) except as a temporary last resort on a network you trust — and turn it back on afterwards.
- Simplest: install from another network before you travel, and not on the morning of the first session.

### `Address already in use` / port 8888

JupyterLab is already running, possibly in a terminal you forgot. Go back to it, or use another port:

```bash
jupyter lab --port 8889
```

To see what is running: `jupyter server list`. The same command prints the `?token=...` address again if you lost it.

### Plots do not appear

Add `%matplotlib inline` at the top of the cell. If you installed `ipympl` and want zoomable figures, use `%matplotlib widget` instead — but this needs working widgets, so fix [that](#widgets-do-not-display) first. In a `.py` script you need `plt.show()`.

### The kernel keeps dying or restarting

Usually memory. Close other notebooks — each one holds its own kernel and its own copy of your data — and avoid creating huge arrays by accident, such as `np.linspace(0, 1, 1e9)`. If it happens at import time on a fresh install, the environment is broken: rebuild it (below).

### Windows: strange failures with paths

If your Windows user name contains spaces or accents, reinstall Miniforge to a simple path such as `C:\miniforge3`, and keep the course folder somewhere simple too. Deeply nested folders can hit the old 260-character path limit. Antivirus software can also make conda very slow; if possible, whitelist the Miniforge folder.

### Starting over

Only your own notebooks matter. Deleting and rebuilding the environment takes a few minutes and fixes many problems:

```bash
conda deactivate
conda env remove -n cas
conda create -n cas -c conda-forge python=3.12 numpy scipy matplotlib pandas seaborn ipywidgets ipympl jupyterlab notebook
conda activate cas
```

`conda env list` shows what you have; `conda clean --all` frees disk space.

### Still stuck?

Come to the school anyway and find a tutor before or during the first session. Please bring: your operating system and version, the output of `conda --version` and `conda env list`, the [one-cell self-check](#the-one-cell-self-check), and the **complete** error message — the last line alone is rarely enough. As a last resort you can follow the course on [Binder](https://mybinder.org/v2/gh/cerncas/hands-on-lattice-exercises/HEAD), remembering to download your notebooks before closing the tab.

---

# Appendix A: Python packages and cheatsheets

Below are the packages most useful for our course — concentrate on `numpy` and `matplotlib` — and a few popular extras.

### The *numpy* package

Have a look at this [summary poster](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf). It covers the instructions you should be familiar with.

[![numpy cheatsheet](_img_instructions/upload_6ffb4d07b1ebb895528f2a34aae41ec6.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)

The official [numpy beginner's guide](https://numpy.org/doc/stable/user/absolute_beginners.html) is also excellent and always up to date.

### The *matplotlib* package

[![matplotlib cheatsheet](_img_instructions/upload_4b54812812e21978b600b860ba1ddf5b.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)

See also the official [matplotlib cheatsheets](https://matplotlib.org/cheatsheets/) and the [example gallery](https://matplotlib.org/stable/gallery/index.html) — the fastest way to make a plot is to find one you like there and copy its code.

### The *linalg* module

[![scipy linear algebra cheatsheet](_img_instructions/upload_15561fc12184bb0ae3f9cf7b1850317a.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_SciPy_Cheat_Sheet_Linear_Algebra.pdf)

### The *pandas* package (optional)

[![pandas cheatsheet](_img_instructions/upload_90383c01e29d29fb6a5516c613e22c4d.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)

### The *seaborn* package (optional)

[![seaborn cheatsheet](_img_instructions/upload_9a3c3f5ca48bbd567a0662df20dbd16f.png)](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf)

### The *sympy* package (optional)

[![sympy cheatsheet](_img_instructions/upload_fc7a06ea6135d2bf17311bd7a91f1a9f.png)](http://daabzlatex.s3.amazonaws.com/9065616cce623384fe5394eddfea4c52.pdf)

---

# Appendix B: A minimal terminal survival kit

If the terminal is new to you, these six commands are enough for the whole setup.

| What you want | Linux / macOS / WSL | Windows (Miniforge Prompt) |
|---|---|---|
| Where am I? | `pwd` | `cd` |
| What is in here? | `ls` | `dir` |
| Go into a folder | `cd foldername` | `cd foldername` |
| Go up one level | `cd ..` | `cd ..` |
| Go to my home folder | `cd ~` | `cd %USERPROFILE%` |
| Stop a running program | `Ctrl`+`C` | `Ctrl`+`C` |

Three habits that save time:

- **`Tab` completes names.** Type `cd hands`, press `Tab`, and the shell finishes it. This also avoids typos.
- **The up arrow recalls previous commands.** You will type `conda activate cas` often.
- **Paths with spaces need quotes:** `cd "My Documents"`.

To open a terminal directly in a folder: on Windows, type `cmd` in the address bar of File Explorer; on macOS, right-click the folder → *Services → New Terminal at Folder*; on most Linux desktops, right-click → *Open in Terminal*.

---

# Appendix C: A minimal conda survival kit

| Task | Command |
|---|---|
| List all environments | `conda env list` |
| Activate an environment | `conda activate cas` |
| Leave an environment | `conda deactivate` |
| List packages in the active environment | `conda list` |
| Check a single package | `conda list numpy` |
| Install a package | `conda install -c conda-forge <package>` |
| Build an environment from a file | `conda env create -f environment.yml` |
| Update an environment from a file | `conda env update -f environment.yml --prune` |
| Export what you have | `conda env export > my_environment.yml` |
| Delete an environment | `conda env remove -n cas` |
| Free disk space | `conda clean --all` |

Two rules worth remembering:

1. **Never install into `base`.** Keep it as the minimal launcher it is meant to be, and give every project its own environment. A damaged `base` means reinstalling everything; a damaged `cas` is rebuilt in three minutes.
2. **Prefer `conda install` over `pip install` inside a conda environment.** Mixing them can break things in confusing ways. If a package exists only on pip, install everything else with conda *first*, then use pip, and do not go back to conda for that environment.

---

*This document is maintained in the [cerncas/hands-on-python](https://github.com/cerncas/hands-on-python) repository. Corrections and reports of anything that did not work on your machine are very welcome — this is how the guide improves each year.*