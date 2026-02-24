# How to set up the communication channel ?

0) Adapt the paths in the used files to your personnal environment
1) Upload the sketches on the respective LoRa arduino modules.
2) Connect the sender to the Rasp Pi through I2C
3) Connect the receiver to the Rasp Pi through Serial
4) Run these programs :
    1) on **RASP PI** : python real_time_data_monitoring.py (send data through I2C as it is being written)
    2) on **PC** : python serial_receiver.py (listen to Serial data and store it into a csv file)
    3) on **PC** : node server.js (backend : send data to the html server as it is being written in the csv file)
    4) on **PC** : xdg-open internest.html (frontend)
    5) (for simulation only) on **RASP PI** : python test_csv.py (start data simulation and csv file writing)

