
import pygame
import datetime
pygame.init()
time1 = datetime.datetime(2020, 12, 31, 0, 0, 0)
for i in range(30):
    time1 += datetime.timedelta(minutes=6)
    print(time1.strftime("%Hh %Mm")[1:])
    text = time1.strftime("%Hh %Mm")[1:]
    #设置字体和字号
    font = pygame.font.SysFont('Microsoft YaHei', 64)
    #渲染图片，设置背景颜色和字体样式,前面的颜色是字体颜色
    ftext = font.render(text, True, (65, 83, 130),(255, 255, 255))
    #保存图片
    pygame.image.save(ftext, 'C:/Users/Ailab/Desktop/terry/序号/小时分钟/'+str(i+1)+".jpg")#图片保存地址