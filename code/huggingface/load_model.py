from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aic-sejong-team/act_AIC",
    local_dir="../../aic_data/model/act_AIC",
)