# Gait Identifier: Plan

A wearable made from an **Arduino UNO Q** and an **MPU-6050** IMU that tells
apart four activities:

| Label | What you're doing |
|---|---|
| `still` | standing still |
| `walk` | walking |
| `run` | running |
| `bike` | biking, or lying on your back and air-biking |

There are two models, and neither uses a neural net:

1. **Peak model:** uses only one number per window, the peak acceleration
   along one axis.
2. **Full model:** uses everything the IMU measures (3-axis accelerometer and
   3-axis gyroscope).

All the logic is in Python. Only a short Arduino sketch reads the sensor.

> **Assumption:** "5060" is read here as the **MPU-6050** (6-axis accel + gyro,
> I2C address 0x68). If it's a different part, only the sketch's sensor-reading
> code changes; the rest of the plan stays the same.

---

## 1. Hardware and where to wear it

```
 MPU-6050 --(I2C / Qwiic)--> UNO Q microcontroller (sketch: read at 100 Hz)
                                   |  Bridge (RPC)
                                   v
                          UNO Q Linux side (Python: features + models)
                                   |
                     LED matrix (shows activity) + MQTT (laptop dashboard)
```

- **Wiring:** MPU-6050 VCC to 3.3V, GND to GND, SDA/SCL to the UNO Q's
  I2C pins or its Qwiic connector.
- **Sensor settings:** set accel range to **±8 g** and gyro range to
  **±1000 °/s**, because running hits several g. Use the built-in low-pass
  filter at about 44 Hz.
- **Placement: strapped to the front of the thigh**, just above the knee,
  with the sensor's +X axis pointing along the thigh toward the knee. The
  thigh works better than the wrist or a pocket for this:
  - Biking moves the thigh in big smooth circles but has no foot-strike
    jolt. Walking and running have a jolt every step.
  - When you're lying on your back to air-bike, gravity points a different
    way relative to the thigh, which helps tell it apart from standing.
  - Always mount it the same way. Mark "this side out, arrow to knee" on the
    strap.
- **Power:** the UNO Q runs Linux by itself, so a USB-C power bank in a
  pocket or armband powers the whole thing with no laptop attached.
  Put the board and battery in a small case or sleeve.

## 2. Software layout (`gait-identifier/`)

```
gait-identifier/
  PLAN.md                  this file
  unoq_app/                  Arduino App Lab app (runs on the UNO Q)
    sketch/sketch.ino        read MPU-6050 at 100 Hz into a ring buffer
    python/main.py           record mode + live classify mode
    python/features.py       window -> feature vector (shared with laptop)
    python/models.py         peak model + full model, loaded from JSON
  collect.py               laptop: start/stop labelled recordings over MQTT
  train.py                 laptop: build features, train both models, export JSON
  evaluate.py              laptop: cross-validation, confusion matrices, plots
  data/                    recorded CSVs (one file per session)
  models/peak_model.json   trained models (copied onto the UNO Q)
  models/full_model.json
```

- **Train on the laptop, run on the board.** Training uses numpy and
  scikit-learn on the laptop. The trained model is saved as plain JSON
  (thresholds, tree splits, or stored training points), the same idea as
  `ping_pong/swing_model.json`. The UNO Q only needs **numpy** to run it, so
  there's no heavy install on the board.
- **Bridge throughput:** calling the sketch once per sample (100 calls a
  second) is too slow. Instead the sketch keeps a ring buffer, and Python
  calls `Bridge.call("read_batch")` about 10 times a second to get the newest
  ~10 samples at once.

## 3. Data collection

- **Window:** 2 s of data (200 samples at 100 Hz), with a new window every
  0.5 s (75% overlap). 2 s covers at least one full stride or pedal turn even
  when walking slowly.
- **Recording:** run `collect.py walk` on the laptop. It sends
  `start walk` over MQTT to the board, which saves raw samples to
  `data/<label>_<person>_<time>.csv` until it gets `stop`.
  Columns: `t, ax, ay, az, gx, gy, gz`.
- **How much:** per person, about **3 × 1 minute of each activity**,
  recorded at different times and paces (slow/normal/brisk walk;
  jog/fast run; real bike and air-bike; standing still and also
  shifting your weight around). At least 2–3 people if possible.
- **Trimming:** cut the first and last 3 s of each recording, where you're
  starting or stopping.

## 4. Model 1: peak acceleration in one direction

**Feature (just one number):** the largest value of acceleration along a
single chosen axis in the 2 s window, in g.

**Which axis:** pick it from the data. Try each of the six directions
(+X, −X, +Y, −Y, +Z, −Z), measure how well each separates the classes using
cross-validation, and keep the best one. The guess is **+X, along the thigh**:

| Activity | Expected peak along the thigh (rough) | Why |
|---|---|---|
| still | ≈ 1.0 g | gravity points straight along the thigh |
| bike / air-bike | ≈ 0.3–1.1 g | thigh is tilted, so less gravity; smooth motion, no jolts. Lying down makes it lower still |
| walk | ≈ 1.3–2 g | a jolt at every heel strike |
| run | ≈ 2.5–6 g | much bigger jolts |

**Classifier: a 1-feature decision tree**, which on one feature is just a
set of thresholds learned from the data:

```
peak < t1        -> bike
t1 <= peak < t2  -> still
t2 <= peak < t3  -> walk
peak >= t3       -> run
```

As a comparison, also try **KNN on the single feature**, and a version
with hand-set thresholds read off a histogram.

**Expected weakness:** `still` and `bike` both sit near 1 g with no jolts.
If the bike's thigh tilt happens to reach vertical, the two overlap, so
expect most of this model's mistakes to be still↔bike. That's the point of
comparing it with model 2.

## 5. Model 2: all IMU outputs

**Features per window** (about 40 numbers), computed in `features.py` with
numpy:

| Group | Features |
|---|---|
| Per axis (ax, ay, az, gx, gy, gz) | mean, standard deviation, min, max, range |
| Magnitudes | ‖a‖ and ‖g‖: mean, std, max, 90th percentile |
| Rhythm | dominant frequency of ‖a‖ and of gx (FFT) and how strong it is (steps or pedal turns per second) |
| Orientation | average thigh pitch and roll from gravity (separates standing from lying on your back) |
| Smoothness | mean absolute jerk (change in ‖a‖ between samples): high for foot strikes, low for pedalling |
| Coupling | correlation between gyro rotation and acceleration (biking is smooth rotation without impacts) |

**Classifiers to compare:**
- **Decision tree** (depth limited to about 4–5): easy to read and explain.
  Export it to JSON and print the tree to show which features it uses.
- **KNN** (k ≈ 5) on standardized features (each feature scaled to mean 0,
  std 1 using the training data).
- *Optional:* random forest, if the single tree overfits. Still not a
  neural net.

Pick the best one by cross-validation and use it on the board.

## 6. Testing and comparing the two models

- **Leave-one-session-out cross-validation:** train on every recording
  except one, test on that one, and repeat for each. If there are several
  people, also do **leave-one-person-out**. This tests on data the model
  has never seen and doesn't over-count by testing on overlapping windows
  from the same recording.
- **Report for each model:** overall accuracy, a **confusion matrix**
  (which activities get mixed up), and per-class accuracy.
- **Plots:** a histogram of the peak feature coloured by activity (shows
  where model 1's thresholds fall and where classes overlap), and a scatter
  of two strong model-2 features.
- **Expected result:** model 1 is good at still vs walk vs run, but mixes up
  still and bike. Model 2 should fix that using rhythm, smoothness and
  orientation.

## 7. Live demo on the wearable

- Every 0.5 s, the UNO Q classifies the newest 2 s window with **both**
  models.
- **Smoothing:** report the most common answer over the last 3 windows, so
  it doesn't flicker between classes.
- **Output:**
  - The UNO Q's **built-in LED matrix** shows an icon or letter for the full
    model's answer (S / W / R / B).
  - **MQTT** publishes both models' answers to `ME193/Rogers/Fiona/gait`
    (broker `broker.emqx.io`; `test.mosquitto.org` stopped answering from Tufts).
  - A small laptop script shows both answers side by side live.
- **Switch modes** with an MQTT command: `record <label>`, `stop`,
  `classify`, `model peak` / `model full`.

## 8. Milestones

1. **Sensor check:** the sketch reads the MPU-6050 at 100 Hz, and Python on
   the UNO Q prints the samples. Confirm that the sample rate is right and
   that +X points along the thigh (it should read ax ≈ −1 g when standing: an accelerometer at rest reads +1 g along the axis pointing up).
2. **Recording:** `collect.py` + record mode save labelled CSVs. Collect the
   first dataset from one person.
3. **Features + model 1:** `features.py`, pick the axis, threshold tree,
   cross-validation results and histogram.
4. **Model 2:** full features, decision tree and KNN, cross-validation,
   confusion matrices.
5. **On the board:** export JSON, run both models live, LED matrix and
   MQTT output.
6. **More data and tuning:** add more people and sessions, retrain, and
   write up the comparison.

## 9. Risks

| Risk | What to do |
|---|---|
| Data from the sensor arrives too slowly through the Bridge | Read in batches from a buffer on the microcontroller (section 2); drop to 50 Hz if needed |
| Strap rotates on the thigh, changing which way the axes point | Firm strap, marked orientation. Model 2 can also use magnitudes, which don't depend on orientation |
| Running saturates the sensor | ±8 g / ±1000 °/s ranges; check the recorded max |
| Model works for you but not someone else | Record more than one person; test leave-one-person-out |
| Air-biking looks different from real biking | Record both under the `bike` label so the model learns both |
| scikit-learn is hard to install on the board | Only train on the laptop; the board just runs the JSON model with numpy |
