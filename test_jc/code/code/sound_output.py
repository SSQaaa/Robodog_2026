# -*- coding: utf-8 -*-

from playsound import playsound
import time
class sound_output:
    def __init__(self):
        self.red_barrel = 'sound/red_barrel.mp3'
        self.orange_barrel = 'sound/orange_barrel.mp3'
        self.blue_barrel = 'sound/blue_barrel.mp3'
        self.yellow_barrel = 'sound/yellow_barrel.mp3'
        self.number_1 = 'sound/1.mp3'
        self.number_2 = 'sound/2.mp3'
        self.number_3 = 'sound/3.mp3'
        self.number_4 = 'sound/4.mp3'
        self.number_5 = 'sound/5.mp3'
        self.number_6 = 'sound/6.mp3'
        self.low = 'sound/low.mp3'
        self.normal = 'sound/normal.mp3'
        self.high = 'sound/high.mp3'

    def output(self,color,number,status):
        playsound(self.__dict__[color])
        time.sleep(0.1)
        playsound(self.__dict__[f'number_{number}'])
        time.sleep(0.1)
        playsound(self.__dict__[status])
        time.sleep(0.1)




