# SimonAndRandyNotes
Notes for Simon and Randy's collaboration

This has LaTeX code for the notes taken during the collaboration.

$$
\begin{bmatrix}
I   & 0   & 0 \\
f_1 & 0   & 0 \\
0   & f_2 & 0 \\
\end{bmatrix}
$$

## Directory Structure

This repository is organized into the following directories:

- **`latex/`** - LaTeX documents and papers
  - Contains subdirectories for specific projects (e.g., `Training_DS_by_proxy/`)
- **`markdown/`** - Markdown notes from meetings and discussions
  - Date-stamped notes files (e.g., `4-4-2025_notes.md`)
  - Time capsule documents from visits
- **`notebooks/`** - Jupyter notebooks for data analysis and experiments
  - `generated/` - Auto-generated or derived notebooks
- **`lux/`** - Lux-related files and configurations

## Creating a New Directory

To create a new directory in this repository:

1. **Determine the appropriate parent location** based on content type:
   - LaTeX documents → `latex/`
   - Markdown notes → `markdown/`
   - Jupyter notebooks → `notebooks/`
   - Lux content → `lux/`

2. **Create the directory** using one of these methods:

   **Via command line:**
   ```bash
   mkdir -p path/to/new-directory
   ```

   **Via GitHub web interface:**
   - Navigate to the desired parent directory
   - Click "Add file" → "Create new file"
   - In the filename field, type `new-directory-name/README.md`
   - Add content and commit

3. **Add a README or initial file** to describe the directory's purpose (optional but recommended for subdirectories)

4. **Commit and push your changes:**
   ```bash
   git add .
   git commit -m "Add new directory for [purpose]"
   git push
   ```

### Naming Conventions

- Use descriptive names that clearly indicate the directory's purpose
- For subdirectories, follow the existing convention (e.g., `Training_DS_by_proxy/` uses underscores and mixed case)
- For date-based content, use the format `M-D-YYYY` without leading zeros (e.g., `4-4-2025_notes.md`, `4-11-2025_notes.md`)

### Examples

**Creating a new LaTeX project:**
```bash
mkdir -p latex/my-new-paper
cd latex/my-new-paper
touch main.tex
```

**Creating a new notebook collection:**
```bash
mkdir -p notebooks/my-analysis
cd notebooks/my-analysis
touch analysis.ipynb
```
