from time import sleep
import ctypes

print('process started, admin: '+str(ctypes.windll.shell32.IsUserAnAdmin()!=0))
sleep(3)
aa
print('process ended')