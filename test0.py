#!/usr/bin/env python3
import spidev
import time 

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 1350000

def read_channel(ch):
	adc = spi.xfer2([1, (8+ch) << 4, 0])
	return ((adc[1] & 3) << 8) + adc[2]
	
while True: 
	val = read_channel(0)
	print('RAW:', val, 'NORMALIZED', val/1023)
	time.sleep(0.2)

