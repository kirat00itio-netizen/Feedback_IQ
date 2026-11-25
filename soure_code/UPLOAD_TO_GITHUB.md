# Upload instructions (GitHub website)

**Unzip first.** GitHub does not extract a ZIP into a repository.

1. Go to github.com -> New repository -> Create.
2. Select Add file -> Upload files.
3. Upload the contents of the extracted `FeedbackIQ_GitHub_Ready` folder.
4. Commit changes.
5. The project root should contain `README.md`, `requirements.txt`, `src/`, and `app/`.
6. Run locally: `pip install -r requirements.txt` then `streamlit run app/streamlit_app.py`.
7. To publish a running web app, connect this repository to a Streamlit host.

Included original dataset files are under 25 MiB each. GitHub's browser upload
limit applies to *each individual file*, not the sum of all files.

**Never upload** `.env`, model checkpoints, a Python virtual environment (`venv` / `lib`),
or a `.git` folder from another local repository.
