# Breast Ultrasound ML — Web App

Classical + hybrid-quantum breast ultrasound image classifier, built with Streamlit.

## Run it locally

```
pip install -r requirements.txt
streamlit run app/app.py
```

## Put it online so anyone (including on a phone) can open it with a link

This uses **Streamlit Community Cloud**, which is free and hosts the app permanently
at a URL like `https://your-app-name.streamlit.app`. No installs needed on the phone —
just open the link in any mobile browser.

### 1. Create a GitHub repo and push this folder

This folder is already a git repo with an initial commit made. You just need to
create the remote repo on GitHub and push to it.

1. Go to https://github.com/new, create a repo (e.g. `breast-ultrasound-ml-webapp`).
   Public is fine and free; keep it Private only if you're on a paid Streamlit Cloud plan.
2. In this folder, run:
   ```
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   (Replace `<your-username>` and `<repo-name>`.)

### 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **New app**.
3. Pick the repo you just pushed, branch `main`, and set **Main file path** to `app/app.py`.
4. Click **Deploy**. First build takes a few minutes (it installs torch + pennylane).
5. You'll get a permanent URL — share that with your teammate. They open it in their
   phone's browser, no compiling or installing anything.

### Notes

- The model files (`models/*.pt`, ~44 MB total) are committed directly to the repo —
  no Git LFS needed, they're under GitHub's 100 MB per-file limit.
- Any time you push a new commit to `main`, the deployed app auto-updates.
