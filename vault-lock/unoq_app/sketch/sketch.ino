// Finger-code safe: the UNO Q's microcontroller side.
//
// The camera, the neural net and the lock logic all run in Python on the
// Linux side (python/main.py). This sketch is the hands and eyes of the safe:
//
//   external button (D2)     Python asks "any presses?"   Bridge.call("presses")
//   RGB LEDs 3 and 4         Python says what to show     Bridge.call("show", code)
//   LED matrix (13 x 8)        code = mode * 1000 + arg
//   servo (D9)               open (90 deg) only in the UNLOCKED mode
//
// Fail-secure: if Python stops calling "show" for 5 s (crashed, restarting,
// unplugged camera...), the servo goes back to locked and the matrix shows "?".
//
// Wiring
//   Button: one leg to D2, the other to GND (uses the internal pull-up).
//   Servo:  signal (orange/yellow) to D9, red to 5V, brown/black to GND. A big
//           servo should get its own 5V supply (grounds connected), or the
//           board may reset when it moves.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

// 1 = the Servo library. If it doesn't compile for the UNO Q, set 0: the
// sketch then makes the servo pulses itself (works on any pin).
#define USE_SERVO_LIB 1
#if USE_SERVO_LIB
#include <Servo.h>
Servo servo;
#endif

const int BUTTON_PIN = 2;
const int SERVO_PIN = 9;
const int SERVO_LOCKED_DEG = 0;     // same as settings.py
const int SERVO_OPEN_DEG = 90;
const unsigned long WATCHDOG_MS = 5000;

// The on-board RGB LEDs are on when their pin is LOW.
#define LED_ON LOW
#define LED_OFF HIGH
const int LED_PINS[2][3] = {{LED3_R, LED3_G, LED3_B}, {LED4_R, LED4_G, LED4_B}};

// Modes: same numbers as lock.py.
enum { LOCKED = 0, SCANNING = 1, UNLOCKED = 2, DENIED = 3, LOCKOUT = 4, NO_PYTHON = 9 };

Arduino_LED_Matrix matrix;
uint8_t frame[8 * 13];

// Set by the Bridge calls, read by loop().
volatile int mode = NO_PYTHON;
volatile int arg = 0;
volatile unsigned long last_show = 0;
volatile int presses = 0;
volatile int presses_total = 0;   // since power-on, for diagnostics
volatile int button_raw = HIGH;   // D2 right now: HIGH = released, LOW = pressed

int last_entered = 0;
unsigned long flash_until = 0;

// --- Bridge functions (called from Python) ------------------------------------

int show(int code) {
  mode = code / 1000;
  arg = code % 1000;
  last_show = millis();
  return mode;
}

int take_presses() {
  int n = presses;
  presses = 0;
  return n;
}

// For checking the wiring: 1000 * (D2 is LOW, i.e. pressed) + presses since power-on.
int button_diag() {
  return (button_raw == LOW ? 1000 : 0) + presses_total % 1000;
}

// --- Button: count presses (debounced) ------------------------------------------

void read_button() {
  static int stable = HIGH, last_raw = HIGH;
  static unsigned long changed = 0;
  int raw = digitalRead(BUTTON_PIN);
  button_raw = raw;
  if (raw != last_raw) {
    last_raw = raw;
    changed = millis();
  }
  if (raw != stable && millis() - changed > 30) {
    stable = raw;
    if (stable == LOW) {                    // pressed (pulled to GND)
      presses++;
      presses_total++;
    }
  }
}

// --- Servo ---------------------------------------------------------------------

void servo_write(int deg) {
#if USE_SERVO_LIB
  static int now_deg = -1;
  if (deg != now_deg) {
    servo.write(deg);
    now_deg = deg;
  }
#else
  // 50 Hz pulses, 1000 us (0 deg) to 2000 us (180 deg): about 1-2 ms per 20 ms.
  static unsigned long next = 0;
  if ((long)(millis() - next) < 0) return;
  next = millis() + 20;
  int us = map(deg, 0, 180, 1000, 2000);
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(us);
  digitalWrite(SERVO_PIN, LOW);
#endif
}

// --- RGB LEDs ----------------------------------------------------------------------

void led(int which, bool r, bool g, bool b) {
  digitalWrite(LED_PINS[which][0], r ? LED_ON : LED_OFF);
  digitalWrite(LED_PINS[which][1], g ? LED_ON : LED_OFF);
  digitalWrite(LED_PINS[which][2], b ? LED_ON : LED_OFF);
}

// --- LED matrix ---------------------------------------------------------------------
// frame[row * 13 + col], brightness 0..7.

const uint8_t BIG[6][7] = {      // 5x7 digits 0-5, bit 4 = left column
  {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
  {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},
  {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}};
const uint8_t SMALL[10][5] = {   // 3x5 digits 0-9, bit 2 = left column
  {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
  {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7}};

const char *ICON_LOCK[8] = {
  ".....###.....", "....#...#....", "....#...#....", "...#######...",
  "...###.###...", "...###.###...", "...#######...", "............."};
const char *ICON_OK[8] = {
  ".............", "...........#.", "..........#..", ".........#...",
  "..#.....#....", "...#...#.....", "....#.#......", ".....#......."};
const char *ICON_NO[8] = {
  "...#.....#...", "....#...#....", ".....#.#.....", "......#......",
  ".....#.#.....", "....#...#....", "...#.....#...", "............."};
const char *ICON_WHAT[8] = {
  ".....###.....", "....#...#....", "........#....", ".......#.....",
  "......#......", ".............", "......#......", "............."};

void px(int r, int c, uint8_t v) {
  if (r >= 0 && r < 8 && c >= 0 && c < 13) frame[r * 13 + c] = v;
}

void icon(const char *rows[8], uint8_t v) {
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 13; c++)
      if (rows[r][c] == '#') px(r, c, v);
}

void big_digit(int d, int col, uint8_t v) {
  for (int r = 0; r < 7; r++)
    for (int c = 0; c < 5; c++)
      if (BIG[d][r] & (0x10 >> c)) px(r, col + c, v);
}

void small_digit(int d, int row, int col, uint8_t v) {
  for (int r = 0; r < 5; r++)
    for (int c = 0; c < 3; c++)
      if (SMALL[d][r] & (4 >> c)) px(row + r, col + c, v);
}

void render(int m, int a, unsigned long now) {
  memset(frame, 0, sizeof(frame));
  bool blink = (now / 500) % 2;
  switch (m) {
    case LOCKED:
      icon(ICON_LOCK, 2);                                    // dim padlock: waiting
      led(0, 0, 0, 0);
      led(1, 0, 0, 0);
      break;

    case SCANNING: {
      int live = a % 10, entered = (a / 10) % 10, hold = a / 100;   // hold 0..9
      if (live >= 1 && live <= 5) big_digit(live, 0, 7);
      else for (int c = 1; c < 4; c++) px(3, c, 3);           // "-": no number seen
      for (int c = 0; c < 5; c++) px(7, c, c * 9 < hold * 5 ? 5 : 0);   // hold bar
      for (int i = 0; i < 3; i++) {                           // the 3 numbers entered so far
        uint8_t v = i < entered ? 7 : 1;
        px(3 * i, 9, v); px(3 * i, 10, v); px(3 * i + 1, 9, v); px(3 * i + 1, 10, v);
      }
      if (entered > last_entered) flash_until = now + 250;   // flash when a number goes in
      last_entered = entered;
      bool flash = (long)(now - flash_until) < 0;
      led(0, 1, 1, 1);                                        // white: light up the hand
      led(1, !flash, !flash, 1);                              // blue flash on each number
      break;
    }

    case UNLOCKED:
      icon(ICON_OK, 7);
      led(0, 0, 1, 0);
      led(1, 0, 1, 0);
      break;

    case DENIED:
      icon(ICON_NO, 7);
      led(0, 1, 0, 0);
      led(1, 1, 0, 0);
      break;

    case LOCKOUT: {
      int s = min(a, 99);                                     // seconds left
      if (s >= 10) small_digit(s / 10, 1, 3, 6);
      small_digit(s % 10, 1, s >= 10 ? 7 : 5, 6);
      led(0, blink, 0, 0);
      led(1, !blink, 0, 0);
      break;
    }

    default:                                                  // Python isn't talking
      icon(ICON_WHAT, blink ? 3 : 1);
      led(0, 0, 0, 0);
      led(1, 0, 0, 0);
  }
  if (m != SCANNING) last_entered = 0;
  matrix.draw(frame);
}

// --- Setup and loop ---------------------------------------------------------------

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++) {
      pinMode(LED_PINS[i][j], OUTPUT);
      digitalWrite(LED_PINS[i][j], LED_OFF);
    }
#if USE_SERVO_LIB
  servo.attach(SERVO_PIN);
#else
  pinMode(SERVO_PIN, OUTPUT);
#endif
  servo_write(SERVO_LOCKED_DEG);

  matrix.begin();
  matrix.setGrayscaleBits(3);          // brightness 0..7

  Bridge.begin();
  Bridge.provide("show", show);
  Bridge.provide("presses", take_presses);
  Bridge.provide("button_diag", button_diag);
}

void loop() {
  unsigned long now = millis();
  read_button();

  int m = mode, a = arg;
  if (now - last_show > WATCHDOG_MS) m = NO_PYTHON;   // fail-secure

  servo_write(m == UNLOCKED ? SERVO_OPEN_DEG : SERVO_LOCKED_DEG);

  static unsigned long next_draw = 0;
  if ((long)(now - next_draw) >= 0) {                   // redraw at 25 Hz
    next_draw = now + 40;
    render(m, a, now);
  }
}
