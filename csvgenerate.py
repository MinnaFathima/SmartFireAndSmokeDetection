import serial
import csv
import time

# Set up serial communication (update COM port as needed)
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Let Arduino reset

# Open CSV file to write sensor data
with open('sensor_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Temperature', 'Gas', 'LDR', 'Fire'])  # CSV header

    print("Collecting data every 5 seconds. Press Ctrl+C to stop.")

    try:
        while True:
            line = ser.read_until(b'E')  # Read until 'E'
            line = line.decode('utf-8', errors='ignore').strip()

            if line.startswith('A') and 'B' in line and 'C' in line and 'D' in line:
                try:
                    temp = line[line.index('A')+1:line.index('B')]
                    gas = line[line.index('B')+1:line.index('C')]
                    ldr = line[line.index('C')+1:line.index('D')]
                    fire = line[line.index('D')+1:]

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{timestamp} | Temp: {temp} | Gas: {gas} | LDR: {ldr} | Fire: {fire}")

                    writer.writerow([timestamp, temp, gas, ldr, fire])

                except Exception as e:
                    print("Error parsing line:", line)

            time.sleep(5)  # Wait 5 seconds before next reading

    except KeyboardInterrupt:
        print("Data collection stopped.")
        ser.close()
