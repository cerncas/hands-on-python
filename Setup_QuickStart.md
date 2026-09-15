# Setup — Quick Start

**Goal:** get from an empty laptop to a running Jupyter notebook, in about 20 minutes.
Five steps, the same on every operating system except Step 1.

> **This is the short version.** The full [Setup Instructions](./Setup_Instructions.md) are the complete reference, and you should go there if you want:
>
> - **troubleshooting** — a symptom-by-symptom list of what goes wrong and how to fix it;
> - **other ways to install** — Anaconda, WSL (Linux inside Windows), `venv` without conda, graphical installers;
> - **other ways to run the notebooks** — VS Code, JupyterLab Desktop, Anaconda Navigator, Spyder, Binder;
> - **an introduction to Python**, the full set of test examples, and cheatsheets for `numpy`, `matplotlib` and the other packages;
> - the explanation of *why* each step is needed.
>
> If anything below fails, go there first.

---

## Step 1 — Install conda

> **Already have conda?** (Anaconda, Miniconda or Miniforge) — type `conda --version` in a terminal. If it answers with a version number, skip to Step 2.

We use **Miniforge**: a small installer that gives you `conda` plus packages from the free, community-run conda-forge channel.

### Linux / macOS

Open a terminal and run:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh"
```

Accept the licence (`yes`), accept the default location, and answer **`yes`** when asked to initialise your shell.

**Then close the terminal and open a new one.** Your prompt should now start with `(base)`.

### Windows

1. Download **`Miniforge3-Windows-x86_64.exe`** from the [Miniforge releases page](https://github.com/conda-forge/miniforge/releases/latest).
2. Run it. Choose **"Just Me"** and accept the defaults.
   *Exception:* if your Windows username contains spaces or accents (é, ü, ñ…), change the install path to `C:\miniforge3`.
3. Open **"Miniforge Prompt"** from the Start menu.

**Use the Miniforge Prompt for everything below**, not a plain `cmd` window.

---

## Step 2 — Create the course environment

Same command everywhere. Answer `y` when asked to proceed; it takes a few minutes.

```bash
conda create -n cas -c conda-forge python=3.12 numpy scipy matplotlib pandas seaborn ipywidgets ipympl jupyterlab notebook
```

Then activate it:

```bash
conda activate cas
```

Your prompt should now start with `(cas)`.

> **The one rule to remember:** every new terminal starts with no environment active. Always run `conda activate cas` **first**. Most setup problems are just this.

---

## Step 3 — Download the course material

```bash
cd ~/Documents
git clone https://github.com/cerncas/hands-on-lattice-exercises.git
cd hands-on-lattice-exercises
```

On Windows, the first line is `cd %USERPROFILE%\Documents`.

No `git`? Either `conda install -c conda-forge git`, or download the [ZIP](https://github.com/cerncas/hands-on-lattice-exercises/archive/refs/heads/master.zip), unpack it, and `cd` into the unpacked folder.

---

## Step 4 — Launch JupyterLab

From that folder, with `(cas)` active:

```bash
jupyter lab
```

Your browser opens on the JupyterLab interface, listing the course files. Double-click `00_Introduction.ipynb`.

To stop it: press `Ctrl`+`C` twice in the terminal. Closing the browser tab is not enough.

> If no browser opens, copy the `http://localhost:8888/lab?token=...` line from the terminal into your browser — token included.

---

## Step 5 — Check that it works

Create a new Python 3 notebook, paste this into a cell, and press `Shift`+`Enter`:

```python
import sys, platform
import numpy, scipy, matplotlib, pandas, seaborn, ipywidgets
from ipywidgets import IntSlider
from IPython.display import display

print(f"Python      {platform.python_version()}   ({sys.executable})")
print(f"numpy       {numpy.__version__}")
print(f"ipywidgets  {ipywidgets.__version__}")

import numpy as np
M = np.array([[1., 0.], [1., 1.]])
Omega = np.array([[0., 1.], [-1., 0.]])
assert np.allclose(Omega, M.T @ Omega @ M), "Symplecticity check failed (!)"

display(IntSlider(description="drag me"))
print("\nAll good — see you at CAS!")
```

You are ready when **all three** of these are true:

1. The printed path contains `cas` — e.g. `.../envs/cas/bin/python`.
2. A **slider appears and moves** when you drag it.
3. The last line says *All good*.

---

## If one of those three fails

| Symptom | Fix |
|---|---|
| `conda: command not found` | Windows: use the **Miniforge Prompt**. Linux/macOS: open a *new* terminal; if still missing, run `~/miniforge3/bin/conda init` and open another one. |
| `ModuleNotFoundError` | Check your prompt says `(cas)`. Then *Kernel → Restart Kernel* and re-run. |
| The path does **not** contain `cas` | Quit JupyterLab, run `conda activate cas`, then `jupyter lab` again. |
| Slider shows as text like `IntSlider(value=0...)`, or "Error displaying widget" | Same fix as above — this means JupyterLab was launched from a different environment than the kernel. Then hard-reload the page (`Ctrl`/`Cmd`+`Shift`+`R`). |
| Anything else | See the [full Setup Instructions](./Setup_Instructions.md#troubleshooting), which cover many more cases. |

Still stuck? Come to the school anyway and find a tutor before the first session. Bring your operating system version and the **complete** error message. As an emergency fallback you can run everything in the browser via [Binder](https://mybinder.org/v2/gh/cerncas/hands-on-lattice-exercises/HEAD) — but your work is lost when the tab closes, so download your notebooks.