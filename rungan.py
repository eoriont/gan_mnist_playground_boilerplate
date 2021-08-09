from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load model
model = load_model('generator_model_100.h5')

# Some matplotlib styling
fig, ax = plt.subplots(figsize=(12, 12))
axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)
plt.subplots_adjust(left=0.01, bottom=0.5, top=0.99, right=0.99)

# Create the default image for the plot
vector = np.asarray([[0.0 for _ in range(100)]])
X = model.predict(vector)
im = plt.imshow(X[0, :, :, 0], cmap='gray_r')

# The function to be called anytime a slider's value changes
def update(_):
    vector = np.asarray([[slider.val for slider in sliders]])
    X = model.predict(vector)
    im.set_data(X[0, :, :, 0])


# Create all the sliders
sliders = []
axes = []
for i in range(100):
    axx = plt.axes([0.01, 0.5-(0.02*i), 0.48, 0.01], facecolor=axcolor)
    axes.append(axx)
    slider = Slider(
        label=i,
        valmin=0.0,
        valmax=1.0,
        valinit=0.0,
        ax=axx
    )
    sliders.append(slider)
    slider.on_changed(update)

plt.show()
