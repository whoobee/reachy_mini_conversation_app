#!/usr/bin/env bash
# Run on the Hermes host after `hermes update`: checks that the reachy profile still has its
# route, tools and a working model. Exit code 0 = all good.
set -u
export PATH="$HOME/.local/bin:$PATH"
PROFILE_DIR="$HOME/.hermes/profiles/reachy"
KEY=$(grep -E "^API_SERVER_KEY=" "$PROFILE_DIR/.env" | cut -d= -f2-)
ok=0

echo "1) Route (multiplexed gateway, /p/reachy/):"
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" http://127.0.0.1:8642/p/reachy/v1/models)
echo "   GET /p/reachy/v1/models -> $code"; [ "$code" = "200" ] || ok=1

echo "2) Tools on the API server:"
tools=$(reachy tools list --platform api_server 2>&1)
for needed in homeassistant memory reachy_robot; do
  if echo "$tools" | grep -q "$needed"; then echo "   ✓ $needed"; else echo "   ✗ $needed MISSING"; ok=1; fi
done
reachy plugins list 2>&1 | grep -E "^│ homeassistant " | grep -q "enabled" || { echo "   ✗ homeassistant plugin not enabled (reachy plugins enable homeassistant)"; ok=1; }

echo "3) Probe question (read-only):"
answer=$(curl -s -m90 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  http://127.0.0.1:8642/p/reachy/v1/chat/completions \
  -d '{"model":"reachy","messages":[{"role":"user","content":"Câte grade sunt în dormitor? Doar citește."}]}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])" 2>&1)
echo "   $answer"
grep "tool ha_" "$PROFILE_DIR/logs/agent.log" | tail -1 | grep -q "$(date +%Y-%m-%d)" && echo "   ✓ Home Assistant tool used" || { echo "   ✗ no Home Assistant tool call today"; ok=1; }

[ $ok -eq 0 ] && echo "ALL OK" || echo "SOMETHING NEEDS FIXING (see above)"
exit $ok
