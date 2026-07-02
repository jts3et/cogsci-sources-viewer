#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
REPO="${1:-cogsci-sources-viewer}"
OWNER="$(gh api user -q .login)"
TOKEN="$(gh auth token)"
DIR="/Users/justinstec/Folders for Claude Coworker/tools/cogcorpus_site"
cd "$DIR"
gh repo create "$OWNER/$REPO" --public -d "Cognitive-science sources viewer" 2>/dev/null || true
touch .nojekyll
# Keep incremental git history (do NOT rm -rf .git each deploy — orphan force-pushes
# make GitHub Pages builds flaky). Commit on top and fast-forward push.
if [ ! -d .git ]; then git init -q -b main; fi
git checkout -q -B main
git remote remove origin 2>/dev/null || true
git remote add origin "https://x-access-token:${TOKEN}@github.com/$OWNER/$REPO.git"
git add -A
git -c user.email=noreply@local -c user.name=deploy commit -q -m "Deploy $(git rev-list --count HEAD 2>/dev/null || echo 0)" || echo "(no changes to commit)"
# Try a normal push; if histories diverged (first switch away from orphan pushes), reconcile once.
if ! git push -q origin main 2>/dev/null; then
  git fetch -q origin main 2>/dev/null || true
  git push -q origin main --force
fi
git remote set-url origin "https://github.com/$OWNER/$REPO.git"
gh api -X POST "repos/$OWNER/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" >/dev/null 2>&1 \
  || gh api -X PUT "repos/$OWNER/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" >/dev/null 2>&1 || true
echo "DEPLOYED → https://$OWNER.github.io/$REPO/"
