# Data Directory

Store nuScenes raw data, derived feature caches, and metadata exports here. Raw datasets should remain excluded from version control; rely on download scripts or external storage.

## Suggested Contents
- `raw/` pointers or symlinks to the official nuScenes release.
- `processed/` serialized feature tensors for Trajectron++.
- `metadata/` summaries describing manual and algorithmic cluster memberships.

## Next Steps
- Document the local data acquisition process and environment variables for dataset paths.
- Create scripts in `src/` to populate `processed/` artifacts reproducibly.
