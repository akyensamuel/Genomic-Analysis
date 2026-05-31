# LaTeX Project Structure

## File Organization

```
FINAL YEAR PROJECT/
├── main.tex                          # Main document (compiles all chapters)
├── chapters/
│   ├── chapter1_introduction.tex     # Introduction chapter
│   ├── chapter2_methodology.tex      # Methodology chapter
│   └── chapter3_results.tex          # Results and Discussion chapter
├── knust_logo.png                    # University logo (place here)
└── README.md                         # This file
```

## Setup Instructions

1. **Add University Logo:**
   - Place your KNUST logo image file named `knust_logo.png` in the main project directory
   - Supported formats: `.png`, `.jpg`, `.pdf`
   - If using different filename/format, edit line 24 in `main.tex`:
     ```latex
     \includegraphics[width=3cm]{knust_logo.png}
     ```

2. **Create Chapters Directory:**
   - The `chapters/` folder should exist in the same directory as `main.tex`
   - All chapter files are stored here for organization

## How to Compile

### Option 1: Using pdflatex (Command Line)
```bash
cd "FINAL YEAR PROJECT"
pdflatex main.tex
pdflatex main.tex   # Run twice to generate table of contents
```

### Option 2: Using a LaTeX Editor
- **TeXShop** (Mac), **MiKTeX** (Windows), **TeX Live** (Linux)
- **VS Code** with LaTeX Workshop extension
- **Overleaf** (cloud-based): Upload all files and compile online

### Option 3: Windows Batch Script
Create a file `compile.bat` in the project directory:
```batch
@echo off
pdflatex.exe main.tex
pdflatex.exe main.tex
pause
```
Then double-click `compile.bat` to compile.

## Output

After successful compilation, you will have:
- `main.pdf` - The complete compiled document
- `main.log` - Compilation log (for debugging if errors occur)
- `main.aux`, `main.toc` - Auxiliary files (can be deleted)

## Editing Chapters

Each chapter is in a separate file for easy editing:
- **chapter1_introduction.tex**: Background, problem statement, objectives
- **chapter2_methodology.tex**: Dataset, preprocessing, feature selection methods
- **chapter3_results.tex**: Results tables, analysis, discussion

To edit a chapter, open the corresponding `.tex` file and make changes.

## Adding New Chapters

To add Chapter 4 (for classifier evaluation):
1. Create `chapters/chapter4_classifiers.tex`
2. Add the following line to `main.tex` after line 57:
   ```latex
   \input{chapters/chapter4_classifiers}
   \newpage
   ```

## Common LaTeX Commands

- **Compile**: `pdflatex main.tex`
- **Clean temporary files**: Delete `.aux`, `.log`, `.toc` files
- **View PDF**: Open `main.pdf` with any PDF reader

## Troubleshooting

| Error | Solution |
|-------|----------|
| `File not found: chapters/chapter1_introduction.tex` | Ensure chapter files are in `chapters/` folder |
| `File not found: knust_logo.png` | Place logo image in main directory or update line 24 in main.tex |
| TOC not updating | Run pdflatex twice |
| Special characters not displaying | Ensure file encoding is UTF-8 |

---

**Next Steps:** After classifier training, add Chapter 4 with classifier evaluation results.
