import pygame
import time

class sound_output:
    def __init__(self):
        self.red_barrel = './sound/red_barrel.mp3'
        self.orange_barrel = './sound/orange_barrel.mp3'
        self.blue_barrel = './sound/blue_barrel.mp3'
        self.yellow_barrel = './sound/yellow_barrel.mp3'
        self.number_1 = './sound/1.mp3'
        self.number_2 = './sound/2.mp3'
        self.number_3 = './sound/3.mp3'
        self.number_4 = './sound/4.mp3'
        self.number_5 = './sound/5.mp3'
        self.number_6 = './sound/6.mp3'
        self.low = './sound/low.mp3'
        self.normal = './sound/normal.mp3'
        self.high = './sound/high.mp3'
        pygame.init()
        pygame.mixer.init()

    def output(self, color, number, status):
        for key in [color, f'number_{number}', status]:
            pygame.mixer.music.load(self.__dict__[key])
            pygame.mixer.music.play()
            # 等待当前音频播放完毕
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

if __name__ == '__main__':
    SOUND = sound_output()
    SOUND.output('red_barrel', '2', 'normal')