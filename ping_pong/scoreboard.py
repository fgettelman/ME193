"""
Live scoreboard over MQTT: publishes the current number of continuous hits
as a float (e.g. "7.0") whenever it changes, plus a heartbeat.

Check it from another terminal:
    mosquitto_sub -h test.mosquitto.org -t 'ME193/Rogers/#' -v
"""

import random

import paho.mqtt.client as mqtt

import config


class Scoreboard:
    def __init__(self, broker=config.MQTT_BROKER, port=config.MQTT_PORT, topic=config.MQTT_TOPIC):
        self.topic = topic
        self.connected = False
        self.score = 0.0
        self._last_sent = None
        self._last_sent_time = 0.0

        client_id = f"me193-pingpong-{random.randint(0, 1_000_000)}"
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        # connect_async + loop_start: never blocks the game, and paho keeps
        # retrying in its own thread if the broker drops.
        self.client.connect_async(broker, port, keepalive=30)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = not reason_code.is_failure
        print(f"MQTT {'connected' if self.connected else 'failed'}: {reason_code}")
        if self.connected:
            self._send(self.score)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected = False
        print(f"MQTT disconnected: {reason_code}")

    def set_score(self, score, now):
        self.score = float(score)
        if self.score != self._last_sent:
            self._send(self.score, now)

    def tick(self, now):
        if now - self._last_sent_time >= config.MQTT_HEARTBEAT_S:
            self._send(self.score, now)

    def _send(self, score, now=None):
        if now is not None:
            self._last_sent_time = now
        if not self.connected:
            return
        self.client.publish(self.topic, f"{score:.1f}", qos=1)
        self._last_sent = score

    def close(self):
        self._send(self.score)
        self.client.loop_stop()
        self.client.disconnect()
