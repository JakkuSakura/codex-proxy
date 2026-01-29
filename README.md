# codex-proxy

Codex CLI uses a `developer` role for some messages, but custom API providers like Z.AI don't accept it. Their Chat Completions endpoint only recognizes `system`, `user`, `assistant`, and `tool`. When Codex sends `developer`, Z.AI returns error 1214 ("Incorrect role information") and the request fails.

This proxy fixes that by sitting between Codex and the API, converting `developer` to `system` before forwarding the request. Codex doesn't know the difference.

## How it works

The proxy listens on localhost (port 8765 by default), rewrites the role field in the messages array, and forwards everything else unchanged. Responses pass straight through.

## Setting up Z.AI with Codex

If you're starting from scratch, here's how to get Z.AI's GLM models working with Codex:

### 1. Get a Z.AI API key

1. Go to [https://z.ai/manage-apikey/subscription](https://z.ai/manage-apikey/subscription)
2. Sign up and get your API key
3. Export it as an environment variable:

```bash
# Add to ~/.zshrc or ~/.bashrc
export Z_AI_API_KEY="your-api-key-here"

# Then reload your shell
source ~/.zshrc  # or source ~/.bashrc
```

### 2. Install Codex CLI

```bash
npm i -g @openai/codex
```

### 3. Configure Codex for Z.AI

Edit `~/.codex/config.toml`:

```toml
[model_providers.z_ai]
name = "z.ai - GLM Coding Plan"
base_url = "http://localhost:8765"
env_key = "Z_AI_API_KEY"
wire_api = "chat"

[profiles.glm_4_6]
model = "glm-4.6"
model_provider = "z_ai"
```

The key is setting `base_url` to `http://localhost:8765` (the proxy) instead of the direct Z.AI API URL.

### 4. Install and start the proxy

See the Installation section below.

### 5. Test it

```bash
codex -p glm_4_6 'hello world'
```

If it works without the "Incorrect role information" error, you're all set.

## Installation

Clone the repo and set up the systemd service:

```bash
# Copy the service file
cp systemd/codex-proxy.service ~/.config/systemd/user/

# Edit the TARGET_API_URL in proxy.py if you're using a different endpoint
# Then enable and start
systemctl --user daemon-reload
systemctl --user enable --now codex-proxy.service
```

## Configuration

Update your Codex config (`~/.codex/config.toml`) to point at the proxy:

```toml
[model_providers.z_ai]
name = "z.ai - GLM Coding Plan"
base_url = "http://localhost:8765"
env_key = "Z_AI_API_KEY"
wire_api = "chat"
```

If you need a different port, edit `PROXY_PORT` in `proxy.py`.

## Service management

```bash
# Check status
systemctl --user status codex-proxy.service

# Restart
systemctl --user restart codex-proxy.service

# Follow logs
journalctl --user -u codex-proxy.service -f

# Stop auto-start at boot
systemctl --user disable codex-proxy.service
```

## Testing

```bash
codex -p glm_4_6 'hello world'
```

If you don't get the role error, it's working.

## Troubleshooting

Still getting error 1214?

1. Verify the proxy is running (`systemctl --user status codex-proxy.service`)
2. Confirm `base_url` in your Codex config is `http://localhost:8765`, not the original API URL
3. Check that `Z_AI_API_KEY` is set (`echo $Z_AI_API_KEY`)
4. Check proxy logs for errors (`journalctl --user -u codex-proxy.service -n 50`)

## Uninstall

```bash
systemctl --user stop codex-proxy.service
systemctl --user disable codex-proxy.service
rm ~/.config/systemd/user/codex-proxy.service
# Revert base_url in ~/.codex/config.toml to the original API URL
```
