import can
import struct


# bustype = 'socketcan_native'
# channel = 'can3'
# bitrate = '500000'

class PythonSocketCAN():
    def __init__(self, channel, bustype, bitrate):
        self.bus = can.interface.Bus(channel=channel, bustype=bustype, bitrate=bitrate)
        self.BmAng = None
        self.AmAng = None
        self.BkAng = None
        self.SwAng = None

    # can_producer(0x0CFF0461, 100, 0.01,-180,150, 0.01, -180, 130, 0.01, -180, 10, 0.01, -180)
    # addr : 0CFF0461 ,priority=C(3), PGN=FF04, source=61
    def can_producer(self, addr, data1, data2, data3, data4):

        factor1 = factor2 = factor3 = factor4 = -180
        offset1 = offset2 = offset3 = offset4 = 0.01

        # data convert
        ndata1 = self.can_dataconvertor_send(data1, factor1, offset1)
        ndata2 = self.can_dataconvertor_send(data2, factor2, offset2)
        ndata3 = self.can_dataconvertor_send(data3, factor3, offset3)
        ndata4 = self.can_dataconvertor_send(data4, factor4, offset4)

        hexd1 = struct.pack('<H', ndata1)[0]
        hexd2 = struct.pack('<H', ndata1)[1]
        hexd3 = struct.pack('<H', ndata2)[0]
        hexd4 = struct.pack('<H', ndata2)[1]
        hexd5 = struct.pack('<H', ndata3)[0]
        hexd6 = struct.pack('<H', ndata3)[1]
        hexd7 = struct.pack('<H', ndata4)[0]
        hexd8 = struct.pack('<H', ndata4)[1]

        msg = can.Message(arbitration_id=addr, data=[hexd1, hexd2, hexd3, hexd4, hexd5, hexd6, hexd7, hexd8],
                          extended_id=True)

        # can send
        self.bus.send(msg)

    def can_receive(self):
        try:
            while True:
                msg = self.bus.recv()
                if msg is not None:
                    print(msg.arbitration_id)
                    if msg.arbitration_id == 218038721:
                        Bmbuf1 = struct.unpack('<8B', msg.data)[3]
                        Bmbuf2 = struct.unpack('<8B', msg.data)[4]
                        Bmbuf3 = struct.unpack('<8B', msg.data)[5]
                        Bmbuf = (Bmbuf1 << 0 | Bmbuf2 << 8 | Bmbuf3 << 16)
                        self.BmAng = self.can_dataconvertor_recv(Bmbuf, 1 / 31768, -250)
                        print(self.BmAng)
                    elif msg.arbitration_id == 218038722:
                        Ambuf1 = struct.unpack('<8B', msg.data)[3]
                        Ambuf2 = struct.unpack('<8B', msg.data)[4]
                        Ambuf3 = struct.unpack('<8B', msg.data)[5]
                        Ambuf = (Ambuf1 << 0 | Ambuf2 << 8 | Ambuf3 << 16)
                        self.AmAng = self.can_dataconvertor_recv(Ambuf, 1 / 31768, -250)
                        print(self.AmAng)
                    elif msg.arbitration_id == 218038723:
                        Bkbuf1 = struct.unpack('<8B', msg.data)[3]
                        Bkbuf2 = struct.unpack('<8B', msg.data)[4]
                        Bkbuf3 = struct.unpack('<8B', msg.data)[5]
                        Bkbuf = (Bkbuf1 << 0 | Bkbuf2 << 8 | Bkbuf3 << 16)
                        self.BkAng = self.can_dataconvertor_recv(Bkbuf, 1 / 31768, -250)
                        print(self.BkAng)
                    elif msg.arbitration_id == 218039236:
                        Swbuf = struct.unpack('<4H', msg.data)[0]
                        self.SwAng = self.can_dataconvertor_recv(Swbuf, 0.1, 0)
                        print(self.SwAng)
                    else:
                        pass

        except KeyboardInterrupt:
            pass

    def can_dataconvertor_send(self, data, factor, offset):
        rawvalue = (data - offset) / factor
        return int(rawvalue)

    def can_dataconvertor_recv(self, data, factor, offset):
        physicalvalue = data * factor + offset
        return physicalvalue

# can_producer(0xCFF0461, 100, 0.01,-180,150,0.01,-180,130,0.01,-180,10,0.01,-180)
# can_producer(0xCFF0561, 100, 0.01,-180,150,0.01,-180,130,0.1,-2880,10,0.01,-180)
# can_receive()
