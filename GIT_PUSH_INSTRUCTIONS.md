# Git Push Instructions

Since git is not available in the current environment, run these commands manually:

## Initial Setup

```bash
# Initialize git repository (if not already done)
git init

# Add remote repository
git remote add origin https://github.com/btranTFT/LowLightMini1.git

# Or if remote already exists, update it:
git remote set-url origin https://github.com/btranTFT/LowLightMini1.git
```

## Add and Commit Files

```bash
# Stage all files (large datasets are excluded by .gitignore)
git add .

# Commit changes
git commit -m "Initial commit: Dataset curation and preprocessing for low-light image enhancement"
```

## Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## If you need to authenticate:
- Use GitHub CLI: `gh auth login`
- Or use personal access token when prompted for password
- Or use SSH: `git remote set-url origin git@github.com:btranTFT/LowLightMini1.git`

