# IoT basics (starter note — replace or add your own documents)

## MQTT
MQTT is a lightweight publish/subscribe messaging protocol used by many IoT devices.
- Default ports: 1883 (plain TCP) and 8883 (MQTT over TLS).
- A broker (for example Eclipse Mosquitto) receives messages and forwards them to subscribers.
- Topics are paths like `home/livingroom/temperature`. `+` matches one level, `#` matches all remaining levels.
- Common weakness: brokers that allow anonymous connections. In Mosquitto 2.x anonymous access is off by default; set `allow_anonymous false` and use `password_file` explicitly.

## CoAP
CoAP is a web-like protocol for constrained devices. It runs over UDP, default port 5683 (5684 with DTLS).

## Securing IoT devices
1. Change default credentials before connecting a device to the network.
2. Put IoT devices on a separate VLAN or guest network.
3. Update firmware regularly; disable unused services such as Telnet (port 23) and UPnP.
