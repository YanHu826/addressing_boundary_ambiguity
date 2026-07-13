BRACE KBS submission package

Contents:
- manuscript/BRACE_KBS_manuscript.pdf: final compiled manuscript PDF.
- cover_letter/BRACE_KBS_cover_letter.pdf: cover letter PDF for Knowledge-Based Systems.
- cover_letter/Cover Letter.tex: cover letter LaTeX source.
- highlights/BRACE_highlights.txt: optional editable Highlights file for Elsevier submission.
- reproducibility/: lightweight run provenance, split hashes, and per-case metric CSVs.
- source/: LaTeX source package for the manuscript. The canonical source file is
  source/cas-dc-template_BRACE_fixed.tex.

To rebuild the manuscript from source:
1. Enter the source directory.
2. Run: latexmk -pdf -interaction=nonstopmode -halt-on-error cas-dc-template_BRACE_fixed.tex

The source directory includes the CAS class/style files, bibliography, BST file,
final figures, and required thumbnail assets used by the Elsevier CAS template.

Author confirmation before submission:
- Replace the Funding statement if specific grant information should be declared.
- Confirm the generative-AI declaration reflects the actual manuscript-preparation workflow.
