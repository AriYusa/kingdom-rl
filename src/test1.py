import pygetwindow
import time
import os
import pyautogui
import PIL

# get screensize
x,y = pyautogui.size()
print(f"width={x}\theight={y}")

x2,y2 = pyautogui.size()
x2,y2=int(str(x2)),int(str(y2))
print(x2//2)
print(y2//2)

# also able to edit z3 to specified window-title string like: "Sublime Text (UNREGISTERED)"
window = pygetwindow.getWindowsWithTitle("Kingdom")[1]
# quarter of screen screensize
x3 = x2 // 2
y3 = y2 // 2
window.resizeTo(x3, y3)
# top-left
window.moveTo(0, 0)
window.activate()
time.sleep(1)

# save screenshot
p = pyautogui.screenshot()
p.save('p.png')

# edit screenshot
im = PIL.Image.open('p.png')
im_crop = im.crop((0, 0, x3, y3))
im_crop.save('p.png', quality=100)

# close window
time.sleep(1)
# window.close()