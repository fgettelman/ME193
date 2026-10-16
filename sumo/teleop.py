"""
Keyboard teleop for the sumo robot: press a key, publish a command over MQTT.

    c or SPACE  charge  (full speed forward for a moment; hold to keep going)
    o           out     (we got pushed out: stop and play the sad song)
    r           reset   (back in the ring for the next round)
    q           quit

Run in a second terminal while sumo.py is running:
    python teleop.py
"""

import sys
import termios
import tty

import paho.mqtt.client as mqtt

import config

KEYS = {"c": "charge", " ": "charge", "o": "out", "r": "reset"}


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Could not connect to {config.MQTT_BROKER}: {reason_code}\r")
        return
    client.subscribe(config.TOPIC_STATE)
    print(f"Connected to {config.MQTT_BROKER}.\r")


def on_message(client, userdata, msg):
    print(f"  robot: {msg.payload.decode(errors='replace')}\r")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(config.MQTT_BROKER, config.MQTT_PORT, keepalive=30)
    client.loop_start()

    print("c/SPACE = charge   o = out   r = reset   q = quit")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)          # read one key at a time, no Enter needed
        while True:
            key = sys.stdin.read(1).lower()
            if key == "q":
                break
            cmd = KEYS.get(key)
            if cmd:
                client.publish(config.TOPIC_CMD, cmd, qos=1)
                print(f"sent: {cmd}\r")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
