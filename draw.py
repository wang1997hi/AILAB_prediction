import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
# define colormap

def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF','#C0C0FE','#7A72EE','#1E26D0','#A6FCA8',
                                                                 '#00EA00','#10921A','#FCF464','#C8C802','#8C8C00',
                                                                 '#FEACAC','#FE645C','#EE0230','#D48EFE','#AA24FA'], 80)
def draw_radar(input_path,output_path):
    # img = np.load(input_path)
    img = np.random.randint(0,80,(100,100))
    plt.axis('off')
    plt.imshow(img, cmap=colormap())
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)

draw_radar('','test.png')