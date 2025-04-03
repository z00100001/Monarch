# Monarch — Locally hosted AI Mental Health Pattern Detection

> **Monarch** is a privacy-first, open-source, offline-capable NLP platform designed to detect signs of depression and emotional distress in text.  

---

## The Mission
> I want this project to become something that actually helps people. Not just another AI demo or class project—I want it to feel like something that could exist in the real world and make a difference. The goal isn’t to create the most complex model ever—it’s to create something powerful, usable, and accessible. A tool that someone’s parent could use, or a school counselor, or even someone just trying to understand themselves better. I want it to run entirely offline, with zero data collection, because privacy is everything when it comes to mental health. I want it to feel like it respects the user, not like it's spying on them. And I want to keep learning through this. I don’t know everything about NLP or machine learning yet, but I’m learning fast, and I want this to be the kind of project that pushes me way beyond just “student-level” work. If this thing turns into a publishable research piece, a product, or even a tool that a few people find useful, that's all I could ask for. I just don’t want it to be something I did for a grade. I want it to actually matter.

---

---

> ** What’s the Problem?**  
> People often show early signs of depression or emotional distress through their writing—texts, journal entries, even social media posts.  
> But these signs usually go unnoticed.  
> There’s no simple, offline, or privacy-first tool to detect them. Most existing tools are cloud-based, invasive, or hard to use.

> ** Why Does It Matter?**  
> Early detection can save lives.  
> Many people don’t ask for help—but their language may already be showing they’re struggling.  
> Parents, counselors, and individuals need something ethical, easy, and local that works without risking privacy or requiring technical skill.

> ** How Are We Solving It?**  
> We’re building **Monarch**—a fully offline, open-source AI tool that detects emotional distress through text.  
> It uses pre-trained NLP models (like BERT and VADER) to flag depressive patterns and emotional shifts.  
> Monarch runs entirely on the user’s device—**no internet, no tracking, no data ever leaves your machine.**  
> It’s simple, private, modular—and designed to help.


---

## Project Goals

- Detect signs of mental health distress in language using pre-trained NLP models (e.g., BERT)
- Provide an easy-to-use, local-only interface for analysis
- Export results in human-readable format (PDF, HTML, CSV)
- Be accessible for parents, teachers, therapists, and developers
- Stay modular, transparent, and completely privacy-respecting

---

## Technologies we are using

- Python 3.10+
- Hugging Face Transformers (DistilBERT, RoBERTa)
- PyTorch
- VADER, spaCy, NLTK
- Streamlit
- Matplotlib, Seaborn
- PyInstaller
- Raspberry Pi compatibility (planned)

---

## Project Layout

### `app/`
User-facing UI built in Streamlit.

- `app.py` – Main Streamlit app file
- `ui_helpers.py` – Custom UI formatting functions
- `style.css` – Optional styling for UI polish

### `engine/`
Core NLP analysis pipeline.

- `analyze.py` – Executes full processing (clean → tokenize → infer)
- `clean.py` – Text cleaning (stopwords, punctuation, emojis, etc.)
- `tokenizer.py` – Handles BERT-compatible tokenization
- `model.py` – Loads trained model and makes predictions
- `config.json` – Editable thresholds and logic config

### `train/`
Everything related to training and dataset building.

- `dataset_builder.py` – Scrapes and prepares raw text from Reddit or other sources
- `preprocess.py` – Cleans and tokenizes data for model ingestion
- `train_model.py` – Fine-tunes transformer models and saves weights
- `monarch_model/` – Contains saved trained model + tokenizer

### `data/`
Local-only data storage.
- `scraper.py` - Scraper for training data
- `raw/` – Unprocessed scraped JSON or CSV data
- `processed/` – Labeled, cleaned `.csv` or `.jsonl` training datasets
- `test_inputs/` – Example `.txt` files to test the final app

### `reports/`
User analysis reports (fully local, never shared online).

- `analysis_log.csv` – Optional log of analyzed sessions
- `exported_reports/` – Saved PDF or HTML reports for users

### `assets/`
C-Day-ready visuals and branding.

- `monarch_logo.png` – Branding/logo
- `poster_graphs/` – Visuals for printed poster
- `architecture_diagram.png` – Architecture or flow diagram

### Root Files

- `README.md` – You're here
- `LICENSE` – MIT license
- `requirements.txt` – All pip dependencies
- `run_app.bat` – Windows launcher
- `run_app.sh` – Mac/Linux launcher
- `.gitignore` – Prevents accidental commits of large files or sensitive data

---

## Privacy and Ethics

- All processing is done **locally on the user's device**
- No data is stored in the cloud or transmitted externally
- Users retain 100% ownership of their data
- Future updates will include offline report encryption and optional anonymized exports

---

## Upcoming Features

- Journaling timeline view (weekly/monthly emotion tracking)
- Raspberry Pi deployable “Wellness Station” mode
- Auto-generated reports with smart summaries
- Expandable API interface for local apps and bots
- Optional anonymized data collection for research

---

## Dev Team Notes

- Each folder is modular and independent
- Use GitHub issues and branches to assign modules
- Commit often with meaningful messages
- Do NOT push large model files or datasets—use `.gitignore`
- Use `run_app.bat` or `run_app.sh` to launch without CLI knowledge

---

MIT License © 2025 Tyler Clanton
