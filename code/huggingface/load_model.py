from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aic-sejong-team/baseline",
    local_dir="../../aic_data/model/baseline",
)