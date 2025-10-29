# timestamp=$(date +%Y%m%d_%H%M%S)
# cd /home/hari/b200/validation/distrbuted_training_tools/
# git add .
# git commit -m "Auto commit at $timestamp"
# git push



cd ~/nebius/distrbuted_training_tools    # go to the repo root
git add .                                # stage ALL changes across subdirectories
git commit -m "Auto commit at $(date)"   # commit (adds current timestamp automatically)
git push origin main                     # push to GitHub
