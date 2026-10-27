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
5. Enable Home Assistant for the profile: `reachy plugins enable homeassistant`.
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
- After changing the robot's MCP tools (a new app version), restart both the robot app and the
  shared gateway so Hermes rediscovers them.
