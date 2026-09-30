# Runs any `legalrag` command inside the GPU container.
#   .\scripts\legalrag.ps1 search "Combien de jours de congés payés ?"
#   .\scripts\legalrag.ps1 ask "Combien de jours de congés payés ?"
# .env (API key, model choice) is passed through when it exists. To reach LM Studio on the
# host from inside the container, use RAG_LLM_BASE_URL=http://host.docker.internal:1234/v1
$root = Split-Path $PSScriptRoot -Parent
$envArgs = @()
if (Test-Path "$root\.env") { $envArgs = @("--env-file", "$root\.env") }

docker run --rm --gpus all `
    -v "${root}:/app" -v hf-cache:/root/.cache/huggingface `
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 -e TRANSFORMERS_VERBOSITY=error `
    @envArgs legal-rag-fr:gpu legalrag @args
