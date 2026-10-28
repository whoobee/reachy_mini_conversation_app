# Hermes `reachy` profile (template)

Agent profile for the `hermes` backend of this app (`REALTIME_BACKEND=hermes`): a
[Hermes Agent](https://hermes-agent.nousresearch.com) profile that answers Reachy's voice
turns, controls Home Assistant, and drives the robot through the app's robot MCP server
(emotions, dances, volume, timers and alarms, weather, sleep).

These files are templates. Home location, Home Assistant devices, IPs and keys are
placeholders; keep the real values only on the Hermes host (and, if you like, in a private
copy that is never pushed).

| File | Goes to (on the Hermes host) |
|---|---|
| `SOUL.md.template` | `~/.hermes/profiles/reachy/SOUL.md` (personality and rules, Romanian) |
| `config.yaml.template` | `~/.hermes/profiles/reachy/config.yaml` |
| `profile.env.example` | `~/.hermes/profiles/reachy/.env` (keys) |
| `Modelfile.qwen3.5-reachy` | local fallback model for Ollama |
| `DEVICE_LIST.example.md` | format for the device list in `SOUL.md` |
| `check_after_update.sh` | run after every `hermes update` |

## Setup

1. Create the profile: `hermes profile create reachy --no-skills`.
2. Copy `config.yaml.template` to the profile and replace `<ROBOT_IP>` with the robot's LAN address.
3. Copy `SOUL.md.template` to `SOUL.md` and fill in:
   - `{{HOME_PLACE}}`: village or town and country, e.g. "orașul X, Germania";
   - `{{TIMEZONE}}`: IANA timezone, e.g. `Europe/Berlin`;
   - `{{DEVICE_LIST}}`: rooms and entity IDs (see `DEVICE_LIST.example.md`).
4. Fill the profile `.env` from `profile.env.example`. `API_SERVER_KEY` and `ROBOT_MCP_TOKEN`
   are random strings (`python3 -c "import secrets; print(secrets.token_hex(32))"`) that must
   match `HERMES_API_KEY` and `ROBOT_MCP_TOKEN` in the robot app's `.env`.
5. Home Assistant goes through the robot, not a Hermes plugin: run [ha-mcp](https://github.com/homeassistant-ai/ha-mcp)
   (e.g. Docker `ghcr.io/homeassistant-ai/ha-mcp:8.7.0 ha-mcp-web` with `HOMEASSISTANT_URL`, an admin
   `HOMEASSISTANT_TOKEN` and a random `MCP_SECRET_PATH`) and set `HA_MCP_URL` in the robot app's `.env`.
   Keep `homeassistant` **out** of this profile's toolsets: Hermes' plugin can call any service and would
   bypass engineering mode. Your main Hermes profile can use ha-mcp directly with full access.
6. Optional local fallback model: `ollama create qwen3.5-reachy -f Modelfile.qwen3.5-reachy`
   (Hermes requires at least 64K context; `num_gpu 99` keeps it fully on the GPU).
7. Restart the shared gateway: `hermes -p default gateway restart`.
8. In the robot app's `.env`:
   ```
   REALTIME_BACKEND=hermes
   HERMES_BASE_URL=http://<HERMES_HOST>:8642/p/reachy/v1
   HERMES_API_KEY=<API_SERVER_KEY from step 4>
   ROBOT_MCP_TOKEN=<ROBOT_MCP_TOKEN from step 4>
   ```
9. Run `./check_after_update.sh` on the Hermes host; it should end with `ALL OK`.

## Home Assistant permissions

- Always allowed: reads (states, logs, traces, history, config) and device control in safe domains
  (lights, switches, covers, fans, climate, media, scenes), except protected entities
  (`HA_PROTECTED_PATTERN`, default `valve|lock|alarm|garage|siren|heater|pump`).
- Everything else (automations, scripts, helpers, entities, restarts...) needs **engineering mode**:
  `set_engineering_mode` by voice, switching off by itself after `ENGINEERING_MODE_TIMEOUT_MIN`.
  The personality asks for a clear "da" before every change.

## Notes and gotchas

- **Model:** `deepseek/deepseek-v4.1-flash` via Nous Portal was the best trade-off we measured
  (correct tool use, good Romanian, ~1.5-2 s to first words, very cheap). `gemini-3.8-flash` was
  reliable but about 2x slower; local `qwen3.5:9b` was fast but often claimed actions it never
  took. Keep `reasoning_effort: minimal`.
- `tools.tool_search.enabled: "off"` hands the robot tools to the model directly. Otherwise Hermes
  hides MCP tools behind `tool_describe`/`tool_call`, adding a round trip per action.
- `auxiliary.title_generation.enabled: false`: with a single local Ollama slot, title generation
  blocked every turn for 30 s.
- **After `hermes update`:** since v0.22 the default gateway serves all profiles under
  `/p/<profile>/` on port 8642 (no separate port or service per profile), and Home Assistant
  moved into the `homeassistant` plugin, which must be enabled per profile. Both broke this
  profile once; `check_after_update.sh` catches them.
- **Robot asleep:** the app's MCP server stops while Reachy sleeps, and Hermes then retries only
  every 5 minutes. Run the relay on the robot so the port never goes away (see `.env.example`):
  ```
  # ~/.config/systemd/user/reachy-mcp-relay.service
  [Service]
  ExecStart=<venv>/bin/python -m reachy_mini_conversation_app.robot_mcp_relay --env <app>/.env
  Restart=always
  [Install]
  WantedBy=default.target
  ```
  then `systemctl --user enable --now reachy-mcp-relay` (and `loginctl enable-linger` so it runs without a login).
- After changing the robot's MCP tools (a new app version), restart both the robot app and the
  shared gateway so Hermes rediscovers them.
