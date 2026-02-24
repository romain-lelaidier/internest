import serial.tools.list_ports
import serial
import matplotlib.pyplot as plt
import os
import csv


class SerialCom:
    def __init__(
        self, chunk_per_file, port_name, max_pos_value, max_cost_value, csv_path
    ):
        self.chunk_per_file = chunk_per_file
        self.port_name = port_name
        self.max_pos_value = max_pos_value
        self.max_cost_value = max_cost_value
        self.csv_path = csv_path

    def show_available_ports(self):
        ports = serial.tools.list_ports.comports()
        print("Currently used ports are : ")
        for port in ports:
            print(f"{port.device}: {port.description}")

    def continuous_listening(self, format="bin"):
        ser = serial.Serial(self.port_name, 9600)
        print(f"Connected to port {self.port_name}")
        os.makedirs("data", exist_ok=True)
        last_packet = b""
        while True:
            if format == "bin":
                last_packet = self.listen_save_bindata(ser, last_packet)
            if format == "csv":
                if not os.path.isfile(self.csv_path):
                    with open(self.csv_path, "w", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(
                            [
                                "time",
                                "x",
                                "y",
                                "z",
                                "cost",
                                "id",
                                "confidence",
                                "start_time",
                                "end_time",
                                "birdnet_id",
                            ]
                        )
                if ser.in_waiting > 0:
                    data = ser.read(32)
                    last_packet = self.listen_save_csvdata(data, last_packet)
        # ser.close()

    def listen_save_bindata(self, ser, last_packet):
        """
        Saves data at save_path while removing duplicates (saves 1 file)
        """
        number_of_files = len(os.listdir("data/"))
        save_path = "data/bin_data" + str(number_of_files + 1) + ".bin"
        print(f"Saving at {save_path}")
        output_file = open(save_path, "ab")
        packet_number = 0  # how many packets were saved
        real_number = 0  # how many packets were received
        while packet_number < self.chunk_per_file:
            if ser.in_waiting > 0:
                data = ser.read(32)
                if data != last_packet:
                    output_file.write(data)
                    packet_number += 1
                    last_packet = data
                    print(f"packet {real_number}")
                    # print(data)
                else:
                    print(f"duplicate detected ! packet_numbber = {real_number}")
                real_number += 1
        print(f"Detected {real_number - packet_number} duplicates")
        print(f"Actually saved {packet_number} packets")
        output_file.close()
        print(f"Data saved to {save_path}")
        print(last_packet)
        return last_packet

    def listen_save_csvdata(self, data, last_packet):
        """
        Saves data at self.csv_path while removing duplicates (saves 1 file)
        """
        with open(self.csv_path, "a", newline="") as output_file:
            writer = csv.writer(output_file)
            packet_number = 0  # how many packets were saved
            real_number = 0  # how many packets were received

            if data != last_packet:
                last_packet = data
                packet_number += 1
                print(f"packet {real_number}")

                new_line = list(self.decode_packet(data))
                # Put to the right format for the HMI
                new_line[1] = (new_line[1][0] / 1000, new_line[1][1] / 1000, new_line[1][2] / 1000)  # tens of meters
                # unwrap position
                new_line = [new_line[0]] + list(new_line[1]) + list(new_line)[2:]
                writer.writerow(new_line)
            else:
                print(f"duplicate detected ! packet_numbber = {real_number}")
            real_number += 1
        print(f"Detected {real_number - packet_number} duplicates")
        print(f"Actually saved {packet_number} packets")
        print(f"Data saved to {save_path}")
        return last_packet

    def file_decode_data(self, bin_file_path):
        """
        Opens bin_file and decode the data in it.
        Decodes exactly like it was encoded (see i2c_master.py).
        Returns a tuple of lists

        Returns
        -------
        - times : list of float
        - position : list 3-tuple of floats
        - cost : list of float
        - id : list of int
        - confidence : list of float
        - start_time : list of float
        - end_time : list of float
        - birdnet_id : list of string
        """
        with open(bin_file_path, "rb") as bf:
            raw_data = bf.read()
            times, position, cost, id = [], [], [], []
            confidence, start_time, end_time, birdnet_id = [], [], [], []

            assert (
                len(raw_data) // 32 == self.chunk_per_file
            ), f"len(raw_data) // 32 = {len(raw_data)} // 32 = {len(raw_data) // 32}, self.chunk_per_file = {self.chunk_per_file} but they should be equal."
            packet_list = [
                raw_data[32 * i : 32 * (i + 1)] for i in range(self.chunk_per_file)
            ]

            for packet in packet_list:
                decoded_packet = self.decode_packet(packet)
                times.append(decoded_packet[0])
                position.append(decoded_packet[1])
                cost.append(decoded_packet[2])
                id.append(decoded_packet[3])
                confidence.append(decoded_packet[4])
                start_time.append(decoded_packet[5])
                end_time.append(decoded_packet[6])
                birdnet_id.append(decoded_packet[6])

            return (
                times,
                position,
                cost,
                id,
                confidence,
                start_time,
                end_time,
                birdnet_id,
            )

    def decode_packet(self, packet):
        """
        Arg : bin packet of size 32 bytes
        Returns : decoded data
        """
        t = int.from_bytes(packet[1:9], byteorder="little") / 1e6
        x = (
            int.from_bytes(packet[9:11], byteorder="little", signed=True)
            / 2**16
            * self.max_pos_value
        )
        y = (
            int.from_bytes(packet[11:13], byteorder="little", signed=True)
            / 2**16
            * self.max_pos_value
        )
        z = (
            int.from_bytes(packet[13:15], byteorder="little", signed=True)
            / 2**16
            * self.max_pos_value
        )
        cost = (
            int.from_bytes(packet[15:17], byteorder="little")
            / 2**16
            * self.max_cost_value
        )
        id = int.from_bytes(packet[17:18], byteorder="little")

        confidence = int.from_bytes(packet[18:19], byteorder="little") / 2**8
        start_time = int.from_bytes(packet[19:22], byteorder="little") / 1e4
        end_time = int.from_bytes(packet[22:25], byteorder="little") / 1e4
        print(packet[25:32])
        try:
            birdnet_id = packet[25:32].decode("ascii")
        except Exception as e:
            print(e)
            birdnet_id = "com_err"

        return (
            t,
            (x, y, z),
            cost,
            id,
            confidence,
            start_time,
            end_time,
            birdnet_id,
        )

    def analyse_data(self, file_path, data="time"):
        (
            times,
            position,
            cost,
            id,
            confidence,
            start_time,
            end_time,
            birdnet_id,
        ) = self.file_decode_data(file_path)

        if data == "time":
            min_time, max_time = times[0], times[-1]
            # plt.ylim([min_time - 1, max_time + 1])
            plt.scatter(list(range(self.chunk_per_file)), times)

        if data == "position":
            plt.figure(figsize=(10, 6))
            for j in range(3):
                plt.scatter(
                    list(range(self.chunk_per_file)),
                    [position[i][j] for i in range(self.chunk_per_file)],
                    label=["x", "y", "z"][j],
                )
                plt.legend()
        if data == "cost":
            plt.scatter(list(range(self.chunk_per_file)), cost)
        if data == "id":
            print(f"id list is : {id}")

        if data == "condidence":
            print("confidence list is : ", confidence)
        if data == "start_time":
            plt.scatter(list(range(self.chunk_per_file)), start_time)
        if data == "end_time":
            plt.scatter(list(range(self.chunk_per_file)), end_time)
        if data == "birdnet_id":
            print("birdnet_id list is : ", birdnet_id)

        if data in ["time", "position", "cost", "start_time", "end_time"]:
            plt.title(data)
            plt.show()


if __name__ == "__main__":
    chunk_per_file = 256
    port_name = "COM17"
    max_pos_value = 100 * 100
    max_cost_value = 1
    csv_path = "monitoring_data.csv"

    save_path = "data/bin_data1.bin"
    Serial_com = SerialCom(
        chunk_per_file, port_name, max_pos_value, max_cost_value, csv_path
    )

    mode = "listen"
    if mode == "listen":
        Serial_com.show_available_ports()
        Serial_com.continuous_listening(format="csv")

    # print(Serial_com.decode_data(save_path))

    data_types = [
        "time",
        "position",
        "cost",
        "id",
        "confidence",
        "start_time",
        "end_time",
        "birdnet_id",
    ]
    for data_type in data_types:
        Serial_com.analyse_data(save_path, data=data_type)
