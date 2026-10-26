// Gait identifier: read the MPU-6050 at 100 Hz on the UNO Q's microcontroller.
//
// loop() samples the IMU into a ring buffer. The Python side (python/main.py)
// calls Bridge.call("read_batch") about 10 times a second and gets every new
// sample since the last call as text:
//
//     "t_ms,ax,ay,az,gx,gy,gz;t_ms,ax,..."      (raw int16 sensor counts)
//
// Python turns counts into g and deg/s (settings.py has the scale factors).
//
// Wiring: MPU-6050 (e.g. Adafruit 3886) on the UNO Q's Qwiic connector (Wire1)
// or the SDA/SCL header pins (Wire). setup() looks on both, at 0x68 and 0x69,
// and Python reports where it was found ("sensor: ..." status over MQTT).
// Mount it on the front of the thigh, +X arrow pointing toward the knee.

#include <Arduino_RouterBridge.h>
#include <Wire.h>

TwoWire *bus = &Wire1;

// Set to 1 to show the current activity on the UNO Q's LED matrix.
// Off by default: the matrix code below is untested, so the rest of the
// sketch still builds if the library's API is different.
#define USE_LED_MATRIX 0

uint8_t mpu_addr = 0x68;
bool have_mpu = false;
String sensor = "MPU-6050 not found on Wire1 or Wire (0x68/0x69): check the cable";
const unsigned long SAMPLE_US = 10000;   // 100 Hz

// Ring buffer: ~2.5 s of samples, so Python can be late without losing data.
const int BUF_N = 256;
struct Sample { uint32_t t; int16_t v[6]; };
Sample buf[BUF_N];
volatile uint32_t head = 0;   // written only by loop()
volatile uint32_t tail = 0;   // written only by read_batch()
volatile uint32_t dropped = 0;

void mpu_write(uint8_t reg, uint8_t val) {
  bus->beginTransmission(mpu_addr);
  bus->write(reg);
  bus->write(val);
  bus->endTransmission();
}

// Look for the MPU-6050 on the Qwiic connector first, then the header pins.
bool find_mpu() {
  TwoWire *buses[2] = {&Wire1, &Wire};
  const char *names[2] = {"Wire1 (Qwiic)", "Wire (header pins)"};
  const uint8_t addrs[2] = {0x68, 0x69};
  for (int b = 0; b < 2; b++) {
    buses[b]->begin();
    buses[b]->setClock(400000);
    for (int a = 0; a < 2; a++) {
      buses[b]->beginTransmission(addrs[a]);
      if (buses[b]->endTransmission() == 0) {
        bus = buses[b];
        mpu_addr = addrs[a];
        sensor = String("MPU-6050 found on ") + names[b] + " at 0x" + String(addrs[a], HEX);
        return true;
      }
    }
  }
  return false;
}

String sensor_info() {
  return sensor;
}

void mpu_setup() {
  mpu_write(0x6B, 0x00);   // PWR_MGMT_1: wake up
  delay(100);
  mpu_write(0x6B, 0x01);   // clock from gyro X PLL (more stable)
  mpu_write(0x1A, 0x03);   // CONFIG: low-pass filter ~44 Hz
  mpu_write(0x19, 0x09);   // SMPLRT_DIV: 1 kHz / (1 + 9) = 100 Hz
  mpu_write(0x1B, 0x10);   // GYRO_CONFIG:  +/-1000 deg/s  (32.8 counts per deg/s)
  mpu_write(0x1C, 0x10);   // ACCEL_CONFIG: +/-8 g         (4096 counts per g)
}

bool mpu_read(int16_t out[6]) {
  bus->beginTransmission(mpu_addr);
  bus->write(0x3B);         // ACCEL_XOUT_H; then accel, temperature, gyro
  if (bus->endTransmission(false) != 0) return false;
  if (bus->requestFrom((int)mpu_addr, 14) != 14) return false;
  int16_t r[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = bus->read();
    uint8_t lo = bus->read();
    r[i] = (int16_t)((hi << 8) | lo);
  }
  out[0] = r[0]; out[1] = r[1]; out[2] = r[2];   // accel x y z
  out[3] = r[4]; out[4] = r[5]; out[5] = r[6];   // gyro x y z (skip temperature)
  return true;
}

String read_batch() {
  String s;
  uint32_t h = head;
  int n = 0;
  while (tail != h && n < 60) {             // cap the reply size
    Sample &p = buf[tail % BUF_N];
    s += String(p.t);
    for (int i = 0; i < 6; i++) { s += ','; s += String(p.v[i]); }
    s += ';';
    tail++;
    n++;
  }
  return s;
}

uint32_t dropped_count() {
  return dropped;
}

#if USE_LED_MATRIX
#include <Arduino_LED_Matrix.h>
Arduino_LED_Matrix matrix;
// One 13x8 letter per activity: S(till) W(alk) R(un) B(ike). 1 = LED on.
const uint8_t LETTERS[4][8][13] = {
  {{0,0,0,0,1,1,1,1,1,0,0,0,0},{0,0,0,1,0,0,0,0,0,0,0,0,0},{0,0,0,1,0,0,0,0,0,0,0,0,0},{0,0,0,0,1,1,1,1,0,0,0,0,0},
   {0,0,0,0,0,0,0,0,1,0,0,0,0},{0,0,0,0,0,0,0,0,1,0,0,0,0},{0,0,0,1,1,1,1,1,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0,0}},
  {{0,0,1,0,0,0,0,0,0,1,0,0,0},{0,0,1,0,0,0,0,0,0,1,0,0,0},{0,0,1,0,0,0,0,0,0,1,0,0,0},{0,0,1,0,0,1,1,0,0,1,0,0,0},
   {0,0,1,0,1,0,0,1,0,1,0,0,0},{0,0,1,1,0,0,0,0,1,1,0,0,0},{0,0,1,0,0,0,0,0,0,1,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0,0}},
  {{0,0,0,1,1,1,1,1,0,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0,0},{0,0,0,1,1,1,1,1,0,0,0,0,0},
   {0,0,0,1,0,0,1,0,0,0,0,0,0},{0,0,0,1,0,0,0,1,0,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0,0}},
  {{0,0,0,1,1,1,1,1,0,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0,0},{0,0,0,1,1,1,1,1,0,0,0,0,0},
   {0,0,0,1,0,0,0,0,1,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0,0},{0,0,0,1,1,1,1,1,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0,0}},
};
#endif

// Python calls this with 0..3 (still, walk, run, bike) or -1 to clear.
bool show(int which) {
#if USE_LED_MATRIX
  uint8_t frame[8 * 13];
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 13; c++)
      frame[r * 13 + c] = (which >= 0 && which < 4) ? LETTERS[which][r][c] : 0;
  matrix.draw(frame);
#endif
  return true;
}

void setup() {
  have_mpu = find_mpu();
  if (have_mpu) mpu_setup();
#if USE_LED_MATRIX
  matrix.begin();
#endif
  Bridge.begin();
  Bridge.provide("read_batch", read_batch);
  Bridge.provide("dropped", dropped_count);
  Bridge.provide("show", show);
  Bridge.provide("sensor_info", sensor_info);
}

void loop() {
  if (!have_mpu) {                 // not plugged in yet: look again every 2 s
    static unsigned long last_try = 0;
    if (millis() - last_try < 2000) return;
    last_try = millis();
    have_mpu = find_mpu();
    if (have_mpu) mpu_setup();
    return;
  }
  static unsigned long next = micros();
  if ((long)(micros() - next) < 0) return;
  next += SAMPLE_US;

  int16_t v[6];
  if (!mpu_read(v)) return;
  if (head - tail >= BUF_N) {   // Python fell behind: buffer full, skip this sample
    dropped++;
    return;
  }
  Sample &p = buf[head % BUF_N];
  p.t = millis();
  for (int i = 0; i < 6; i++) p.v[i] = v[i];
  head++;
}
