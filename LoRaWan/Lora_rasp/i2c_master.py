"""
Encode and send data.
Possible to start simulation.
"""

#  Raspberry Pi Master for Arduino Slave
#  i2c_master_pi.py
#  Connects to Arduino via I2C
  
#  DroneBot Workshop 2019
#  https://dronebotworkshop.com

from smbus import SMBus
import time
from math import sin, cos
from random import randint
import csv

class I2CMaster:
    def __init__(self, addr, bus, max_pos_value, max_cost_value, frequency):
        self.addr = addr
        self.bus = bus
        self.max_pos_value = max_pos_value
        self.max_cost_value = max_cost_value
        self.frequency = frequency
        # Initialise in case no value is specified (this data is so far unrelevant)
        self.cost = 0
        self.id = 0
        self.confidence = 0
        self.start_time = 0
        self.end_time = 0
        self.birdnet_id = 'chatest'

    def safe_round(self, two_power, coef):
        """
        Just the same as round but if round(2 ** two_power * coef) = 2 ** two_power, returns 2 ** two_power - 1
        to avoid overflow.
        
        :param two_power: int, number of bits
        :param coef: value between 0 and 1
        """
        rounded = round(2 ** two_power * coef)
        if rounded >= 2 ** two_power:
            return 2 ** two_power - 1
        if rounded <= - 2 ** two_power + 1:
            return - 2 ** two_power + 1
        return rounded

    def led_test(self):
        """Simple LED switch test"""
        print("Enter 0 or 1 for led blink")
        numb = 1
        while numb == 1:
            ledstate = input(">>>>   ")
            if ledstate == "1":
                self.bus.write_byte(self.addr, 0x1) # switch it on
            elif ledstate == "0":
                self.bus.write_byte(self.addr, 0x0) # switch it on
            else:
                numb = 0
    
    def simulate_raw_data(self, verbose=False):
        """
        Returns a paquet of information : 2-tuple, total 31 bytes
        The data is not yet encoded but in the end it should look like this. (_original_ _format_ to _encoded_ _format_)

        - tracking information : 4-tuple
            - time : float to int_64, 
                8 bytes
            - position : 3-tuple of float to int_16, 
                3 * 2 bytes
            - cost : float to int_16, 
                2 bytes
            - id : int to int_8, 
                1 byte

        - birdnet information : 
            - confidence : float[0,1] to int_8, 
                2 bytes
            - start_time : float to int_64, number of (0.1ms) since the beginning of running 5 min, 
                3 bytes
            - end_time : float to int_64, number of (0.1ms) since the beginning of running 5 min, 
                3 bytes
            - birdnet_id : string to 7-long char list, https://github.com/birdnet-team/BirdNET-Analyzer/blob/main/birdnet_analyzer/eBird_taxonomy_codes_2024E.json
                7 bytes
        
        Notes
        -----
        Possible to compress data
        """
        self.current_time = time.time()
        x = 1000 * cos(self.current_time)
        y = 1000 * sin(self.current_time) ** 2
        z = 2000 * cos(self.current_time + 1)
        self.position = (x, y, z)
        self.cost = sin(x*y*z) ** 2
        self.id = randint(0, 100)

        self.confidence = randint(0, 100) / 100
        self.start_time = self.current_time + randint(0, 20)
        self.end_time = self.current_time - randint(0, 20)
        self.birdnet_id = ['rufwar1', 'blfwar1', 'yebwar1'][randint(0, 2)]

        if verbose:
            print(self.current_time, self.position, self.cost, self.id)
            print(self.confidence, self.start_time, self.end_time, self.birdnet_id)

    def data_to_bytes(self, verbose=False):
        """
        Format data attributes to desired format for transfer
        """
        self.bcurrent_time = round(1e6 * self.current_time).to_bytes(8, byteorder='little')
        x, y, z = self.position
        bx = self.safe_round(15, x / self.max_pos_value).to_bytes(2, byteorder='little', signed=True)
        by = self.safe_round(15, y / self.max_pos_value).to_bytes(2, byteorder='little', signed=True)
        bz = self.safe_round(15, z / self.max_pos_value).to_bytes(2, byteorder='little', signed=True)
        self.bposition = (bx, by, bz)
        self.bcost = self.safe_round(16, self.cost / self.max_cost_value).to_bytes(2, byteorder='little')
        self.bid = self.id.to_bytes(1, byteorder='little')

        self.bconfidence = self.safe_round(8, self.confidence).to_bytes(1, byteorder='little')
        self.bstart_time = round(1e4 * (self.start_time % 300)).to_bytes(3, byteorder='little')
        self.bend_time = round(1e4 * (self.end_time % 300)).to_bytes(3, byteorder='little')
        self.bbirdnet_id = self.birdnet_id.encode('ascii')

        if verbose:
            print(self.bend_time)
            print(self.bbirdnet_id)

    def encode_message(self):
        """
        Returns the concatenated bytes following order :

        time(8)@position[0, 1, 2](2,2,2)@cost(2)@id(1)@confidence(2)@start_time(3)@end_time(3)@birdnet_id(7)
        """
        concatenated_bytes = self.bcurrent_time + self.bposition[0] + self.bposition[1] + self.bposition[2] + self.bcost + self.bid
        concatenated_bytes += self.bconfidence + self.bstart_time + self.bend_time + self.bbirdnet_id
        return concatenated_bytes
    
    def create_ex_csv(self, csv_path, size=600):
        sim_data = [["time", "x", "y", "z", "cost"]]
        for i in range(size):
            current_time = time.time()
            x = 1000 * cos(1000*current_time)
            y = 1000 * sin(1000*current_time) ** 2
            z = 2000 * cos(1000*current_time + 1)
            cost = cos(2*1000*current_time) ** 2
            sim_data.append([current_time, x, y, z, cost])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sim_data)

    def send_lines(self, lines):
        for line in lines:
            line = line.split(',')
            if line[0] == "time":
                continue
            self.current_time = float(line[0])
            self.position = (float(line[1]), float(line[2]), float(line[3]))
            self.cost = float(line[4])
            self.id = int(line[5])
            self.confidence = float(line[6])
            self.start_time = float(line[7])
            self.end_time = float(line[8])
            self.birdnet_id = line[9][:-1]
            print(line, self.birdnet_id)

            self.data_to_bytes()
            encoded_message = self.encode_message()
            self.send_message(encoded_message)
            time.sleep(1 / self.frequency)        
        
    def send_pos_csv(self, csv_path):
        with open(csv_path, mode='r') as f:
            positions = list(csv.reader(f))
            print(positions)
            for line in positions[1:]:
                self.current_time = float(line[0])
                self.position = (float(line[1]), float(line[2]), float(line[3]))
                self.cost = float(line[4])
                self.data_to_bytes()
                encoded_message = self.encode_message()
                self.send_message(encoded_message)
                time.sleep(1 / self.frequency)

    def send_message(self, message):
        """Sends message through I2C"""
        self.bus.write_i2c_block_data(self.addr, 0x00, list(message))
        if not hasattr(self, 'packet_count'):
            self.packet_count = 0
        print(f'message n° {self.packet_count} : {message}  of len {len(message)} was sent !')
        print([message[i] for i in range(len(message))])
            # just in case self.packet_count is not defined (send just one message)

        """
        number_of_packets = len(message) // 32 + 1
        for i in range(number_of_packets):
            self.bus.write_i2c_block_data(self.addr, 0x00, list(message)[32*i:32*(i+1)])
            time.sleep(1e-1)  # time for the packets to be sent (less is not possible : the second packet is lost ! Maybe there's something to be done about it though.)
        print(f'message {message} of len {len(message)} was sent in {number_of_packets} packets!')
        """  # if packets are larger than 32 bytes, which I try to avoid.

    def periodical_send(self):
        """
        Periodically sends message (simulated for now) through I2C.

        :param period: period in seconds (doesn't count algorithm execution time)
        """
        self.packet_count = 0
        while True:
            self.simulate_raw_data()
            self.data_to_bytes()
            encoded_message = self.encode_message()
            self.send_message(encoded_message)
            self.packet_count += 1
            time.sleep(1 / self.frequency)


if __name__ == '__main__':
    addr = 0x8 # bus address
    bus = SMBus(1) # indicates /dev/ic2-1
    max_pos_value = 100 * 100  # cm (precision)
    max_cost_value = 1
    frequency = 5  # Hz

    i2c_master = I2CMaster(addr, bus, max_pos_value, max_cost_value, frequency)
    i2c_master.led_test()

    mode="csv"

    if mode == "one":
        i2c_master.simulate_raw_data(verbose=True)
        i2c_master.data_to_bytes(verbose=True)
        message = i2c_master.encode_message()
        i2c_master.send_message(message)
    if mode == "period":
        i2c_master.periodical_send()
    if mode == "csv":
        csv_path = "test.csv"
        i2c_master.create_ex_csv(csv_path)
        i2c_master.send_pos_csv(csv_path)
