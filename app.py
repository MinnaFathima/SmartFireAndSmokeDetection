import serial
import time

# Replace 'COM3' with your Arduino port (e.g., 'COM4', '/dev/ttyUSB0')
ser = serial.Serial('COM3', 9600)
time.sleep(2)  # wait for Arduino to reset

def read_sensor_data():
    data = {'Temperature': '', 'Gas': '', 'Flame': '', 'Light': ''}
    label = ''
    while True:
        if ser.in_waiting:
            char = ser.read().decode('utf-8', errors='ignore')
            if char == 'A':
                label = 'Temperature'
                data[label] = ''
            elif char == 'B':
                label = 'Gas'
                data[label] = ''
            elif char == 'C':
                label = 'Flame'
                data[label] = ''
            elif char == 'D':
                label = 'Light'
                data[label] = ''
            elif char == 'E':
                # All readings received, print and start fresh
                print(f"Temp: {data['Temperature']} °C, Gas: {data['Gas']}, Flame: {data['Flame']}, Light: {data['Light']}")
                label = ''
            elif label:
                data[label] += char  # Append digit to the corresponding sensor
        time.sleep(5)

try:
    read_sensor_data()
except KeyboardInterrupt:
    print("\nStopped by user")
    ser.close()
