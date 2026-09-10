import time
import legoeducation as le

# 1. Setup the connection information from your Connection Card
card_color = le.LEGO_COLOR_GREEN  # Change to your card's color
card_serial = '0997'              # Change to your card's 4-digit serial number

# 2. Initialize and connect to the Single Motor
singlemotor = le.SingleMotor()
print("Connecting to the single motor...")
singlemotor.connect(card_color=card_color, card_serial=card_serial)

# 3. Check if the connection worked
if not singlemotor.connected:
    print('Error connecting to Single Motor. Make sure it is turned on!')
    exit(1)

print("Connected successfully!")

# 4. Turn the motor clockwise at 50% speed
print("Running motor clockwise...")
singlemotor.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=50)

# Let it spin for 3 seconds
time.sleep(3)

# 5. Stop the motor and disconnect cleanly
print("Stopping motor.")
singlemotor.motor_stop()
singlemotor.disconnect()

print("Done!")
