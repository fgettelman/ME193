"""
Watch the live ping-pong score over MQTT -- run this in a second terminal (or
on another computer) while the game is running:

    python3 watch_score.py               # your topic from config.py
    python3 watch_score.py 'ME193/#'     # everyone in the class

Prints each score as it arrives. Ctrl+C to stop.
"""

import sys
import time

import paho.mqtt.client as mqtt

import config

topic = sys.argv[1] if len(sys.argv) > 1 else config.MQTT_TOPIC


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Could not connect to {config.MQTT_BROKER}: {reason_code}")
        return
    client.subscribe(topic)
    print(f"Connected to {config.MQTT_BROKER}, listening on '{topic}'. Ctrl+C to stop.")
    print("(The game re-sends the score every few seconds, so it shows up soon after it starts.)\n")


def on_message(client, userdata, msg):
    score = msg.payload.decode(errors="replace")
    print(f"{time.strftime('%H:%M:%S')}  {msg.topic}  streak = {score}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(config.MQTT_BROKER, config.MQTT_PORT, keepalive=30)
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopped.")
    client.disconnect()
