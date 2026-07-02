#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
REPO="${1:-cogsci-sources-viewer}"
OWNER="$(gh api user -q .login)"
TOKEN="$(gh auth token)"
DIR="/Users/justinstec/Folders for Claude Coworker/tools/cogcorpus_site"
cd "$DIR"
gh repo create "$OWNER/$REPO" --public -d "Cognitive-science sources viewer" 2>/dev/null || echo "(repo exists, reusing)"
rm -rf .git
git init -q -b main
git add -A
git -c user.email=noreply@local -c user.name=deploy commit -q -m "Deploy viewer"
git remote add origin "https://x-access-token:${TOKEN}@github.com/$OWNER/$REPO.git"
git push -q -u origin main --force
git remote set-url origin "https://github.com/$OWNER/$REPO.git"
gh api -X POST "repos/$OWNER/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" 2>/dev/null \
  || gh api -X PUT "repos/$OWNER/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" 2>/dev/null \
  || echo "(enable Pages manually: Settings → Pages → main /root)"
echo "DEPLOYED → https://$OWNER.github.io/$REPO/"
