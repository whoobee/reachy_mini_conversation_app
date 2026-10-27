Format for `{{DEVICE_LIST}}` in SOUL.md: one line per room, Romanian names for what people say,
Home Assistant entity IDs after `=`. A short, explicit list lets a small model act in one step
instead of searching hundreds of entities. Leave out safety devices (locks, water valves, garage).

```
Birou: lumina = light.office_main; jaluzea = cover.office_blind
Sufragerie: lumina principală = switch.livingroom_main; lampa de podea = light.floorlamp; temperatura = sensor.livingroom_temperature
Dormitor: lumina principală = switch.bedroom_main; temperatura = sensor.bedroom_temperature
Afară: temperatura = sensor.outdoor_temperature
```
