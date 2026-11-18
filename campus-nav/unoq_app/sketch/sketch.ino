// Step counter: read the MPU-6050 at 100 Hz on the UNO Q's microcontroller.
// (Same sketch as gait-identifier; the step counting is all in python/.)
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
uint32_t wakeups = 0;         // times the MPU was found asleep (power blip) and woken

void mpu_write(uint8_t reg, uint8_t val) {
  bus->beginTransmission(mpu_addr);
  bus->write(reg);
  bus->write(val);
  bus->endTransmission();
}

// The UNO Q's I2C buses (from the board's Zephyr overlay):
//   Wire  = header SDA/SCL pins = D20 (SDA, PB11) / D21 (SCL, PB10)
//   Wire1 = Qwiic connector (PD13/PD12, no Arduino pin numbers)
//   Wire2 = A4 (SDA, PC1) / A5 (SCL, PC0)
// If the microcontroller resets in the middle of a read, the MPU-6050 can be
// left holding SDA low, and every transfer after that fails. Clocking SCL a
// few times lets it finish and let go ("bus recovery"). Wire.begin() then
// puts the pins back on I2C.
String line_levels = "";

String recover_bus(int sda, int scl) {
  pinMode(sda, INPUT_PULLUP);
  pinMode(scl, INPUT_PULLUP);
  delayMicroseconds(20);
  bool sda_hi = digitalRead(sda), scl_hi = digitalRead(scl);
  String s = String("SDA ") + (sda_hi ? "high" : "LOW") + ", SCL " + (scl_hi ? "high" : "LOW");
  if (!sda_hi && scl_hi) {
    pinMode(scl, OUTPUT);
    for (int i = 0; i < 16 && !digitalRead(sda); i++) {
      digitalWrite(scl, LOW);  delayMicroseconds(10);
      digitalWrite(scl, HIGH); delayMicroseconds(10);
    }
    pinMode(sda, OUTPUT);                   // a STOP: SDA low -> high while SCL high
    digitalWrite(sda, LOW);  delayMicroseconds(10);
    digitalWrite(sda, HIGH); delayMicroseconds(10);
    pinMode(sda, INPUT_PULLUP);
    pinMode(scl, INPUT_PULLUP);
    s += digitalRead(sda) ? " (was stuck, freed)" : " (stuck, couldn't free)";
  }
  return s;
}

// Look for the MPU-6050 on the Qwiic connector, the header pins, then A4/A5.
bool find_mpu() {
  TwoWire *buses[3] = {&Wire1, &Wire, &Wire2};
  const char *names[3] = {"Qwiic", "header SDA/SCL", "A4/A5"};
  const int sda_pin[3] = {-1, 20, 18}, scl_pin[3] = {-1, 21, 19};
  const uint8_t addrs[2] = {0x68, 0x69};
  line_levels = "";
  for (int b = 0; b < 3; b++) {
    if (sda_pin[b] >= 0)
      line_levels += String(names[b]) + ": " + recover_bus(sda_pin[b], scl_pin[b]) + ". ";
    buses[b]->begin();
    buses[b]->setClock(400000);
    for (int a = 0; a < 2; a++) {
      buses[b]->beginTransmission(addrs[a]);
      if (buses[b]->endTransmission() == 0) {
        bus = buses[b];
        mpu_addr = addrs[a];
        bus->beginTransmission(mpu_addr);    // WHO_AM_I: 0x68 on a real MPU-6050
        bus->write(0x75);
        uint8_t who = 0;
        if (bus->endTransmission() == 0 && bus->requestFrom((int)mpu_addr, 1) == 1) who = bus->read();
        sensor = String("MPU-6050 found on ") + names[b] + " at 0x" + String(addrs[a], HEX)
                 + ", WHO_AM_I 0x" + String(who, HEX) + ". " + line_levels;
        return true;
      }
    }
  }
  sensor = String("MPU-6050 not found on Qwiic, header SDA/SCL or A4/A5. ") + line_levels;
  return false;
}

String sensor_info() {
  return sensor + " (woken after power blips: " + String(wakeups) + ")";
}

void mpu_setup() {
  mpu_write(0x6B, 0x00);   // PWR_MGMT_1: wake up
  delay(100);
  mpu_write(0x6B, 0x01);   // clock from gyro X PLL (more stable)
  mpu_write(0x1A, 0x03);   // CONFIG: low-pass filter ~44 Hz
  mpu_write(0x19, 0x09);   // SMPLRT_DIV: 1 kHz / (1 + 9) = 100 Hz
  mpu_write(0x1B, 0x10);   // GYRO_CONFIG:  +/-1000 deg/s  (32.8 counts per deg/s)
  mpu_write(0x1C, 0x10);   // ACCEL_CONFIG: +/-8 g         (4096 counts per g)
  // The MPU's own FIFO keeps sampling at exactly 100 Hz even when this loop
  // is busy (e.g. sending data to Linux), so no samples go missing.
  mpu_write(0x23, 0x78);   // FIFO_EN: accel x/y/z + gyro x/y/z (12 bytes per sample)
  mpu_write(0x6A, 0x04);   // USER_CTRL: reset the FIFO
  delay(2);
  mpu_write(0x6A, 0x40);   // USER_CTRL: FIFO on
}

// Read n bytes starting at register reg.
bool mpu_read_bytes(uint8_t reg, uint8_t *out, int n) {
  bus->beginTransmission(mpu_addr);
  bus->write(reg);
  if (bus->endTransmission(false) != 0) return false;
  if (bus->requestFrom((int)mpu_addr, n) != n) return false;
  for (int i = 0; i < n; i++) out[i] = bus->read();
  return true;
}

// Move whatever the MPU's FIFO holds into our ring buffer.
void drain_fifo() {
  uint8_t c[2];
  if (!mpu_read_bytes(0x72, c, 2)) return;            // FIFO_COUNT
  int count = (c[0] << 8) | c[1];
  if (count >= 1000) {                                  // overflowed (loop stalled > 0.8 s)
    mpu_write(0x6A, 0x44);                              // reset + keep FIFO on
    dropped += count / 12;
    return;
  }
  int n = count / 12;
  uint32_t now = millis();
  int done = 0;
  while (done < n) {
    int k = min(10, n - done);                          // 120 bytes per read (Wire buffer is 256)
    uint8_t raw[120];
    if (!mpu_read_bytes(0x74, raw, k * 12)) return;     // FIFO_R_W
    for (int j = 0; j < k; j++) {
      if (head - tail >= BUF_N) { dropped++; continue; }
      Sample &p = buf[head % BUF_N];
      p.t = now - (uint32_t)(n - 1 - (done + j)) * 10;  // 10 ms apart, newest = now
      for (int i = 0; i < 6; i++)
        p.v[i] = (int16_t)((raw[j * 12 + 2 * i] << 8) | raw[j * 12 + 2 * i + 1]);
      head++;
    }
    done += k;
  }
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
// One 13x8 letter per activity: S(till) W(alk) R(un) B(ike, unused here). 1 = LED on.
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

// Python calls this with 0..2 (still, walk, run) or -1 to clear.
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
  // Bridge first, so the board can always say what's wrong with the sensor.
  Bridge.begin();
  Bridge.provide("read_batch", read_batch);
  Bridge.provide("dropped", dropped_count);
  Bridge.provide("show", show);
  Bridge.provide("sensor_info", sensor_info);
#if USE_LED_MATRIX
  matrix.begin();
#endif
  have_mpu = find_mpu();
  if (have_mpu) mpu_setup();
}

void loop() {
  if (!have_mpu) {                 // not plugged in yet: look again every 3 s
    static unsigned long last_try = 0;
    if (millis() - last_try < 3000) return;
    last_try = millis();
    have_mpu = find_mpu();
    if (have_mpu) mpu_setup();
    return;
  }
  // After a power blip the MPU-6050 restarts asleep and reads all zeros while
  // still answering on I2C. Check its sleep bit once a second and wake it.
  static unsigned long last_check = 0;
  if (millis() - last_check >= 1000) {
    last_check = millis();
    bus->beginTransmission(mpu_addr);
    bus->write(0x6B);                        // PWR_MGMT_1
    bool ok = bus->endTransmission(false) == 0 && bus->requestFrom((int)mpu_addr, 1) == 1;
    uint8_t pm = ok ? bus->read() : 0xFF;
    if (!ok) {
      have_mpu = false;                      // gone: look for it again
      sensor = "MPU-6050 stopped answering: check the cable";
      return;
    }
    if (pm & 0x40) {                         // SLEEP bit set: it was reset
      mpu_setup();
      wakeups++;
    }
  }

  // Empty the MPU's FIFO every ~20 ms (it holds ~0.85 s, so stalls are fine).
  static unsigned long next = 0;
  if ((long)(millis() - next) < 0) return;
  next = millis() + 20;
  drain_fifo();
}
