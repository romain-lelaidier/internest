from sim_sender import send_signal
from CONFIG import *

if __name__ == '__main__':
    send_signal(server_ip, localPort2, sig_type="sum_sin")