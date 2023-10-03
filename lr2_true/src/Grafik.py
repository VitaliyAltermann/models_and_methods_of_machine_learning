import numpy as np
import matplotlib.pyplot as plt

num_epoch = 50  # число эпох обучения / пересчёта

nOpt=0

if nOpt==1:
    plt.plot(range(0,num_epoch), np.load('lossAdam1.npy'), label='Adam1')
elif nOpt==2:
    plt.plot(range(0, num_epoch), np.load('lossAdam32.npy'), label='Adam32')
elif nOpt==3:
    plt.plot(range(0, num_epoch), np.load('lossMomentum1.npy'), label='Momentum1')
elif nOpt==4:
    plt.plot(range(0, num_epoch), np.load('lossMomentum32.npy'), label='Momentum32')
elif nOpt == 5:
    plt.plot(range(0, num_epoch), np.load('lossNAG1.npy'), label='NAG1')
elif nOpt == 6:
    plt.plot(range(0, num_epoch), np.load('lossNAG32.npy'), label='NAG32')
elif nOpt == 7:
    plt.plot(range(0, num_epoch), np.load('lossRMSprop1.npy'), label='RMSprop1')
elif nOpt == 8:
    plt.plot(range(0, num_epoch), np.load('lossRMSprop32.npy'), label='RMSprop32')
else:
    plt.plot(range(0, num_epoch), np.load('lossAdam1.npy'), label='Adam1')
    plt.plot(range(0, num_epoch), np.load('lossAdam32.npy'), label='Adam32')
    plt.plot(range(0, num_epoch), np.load('lossMomentum1.npy'), label='Momentum1')
    plt.plot(range(0, num_epoch), np.load('lossMomentum32.npy'), label='Momentum32')
    plt.plot(range(0, num_epoch), np.load('lossNAG1.npy'), label='NAG1')
    plt.plot(range(0, num_epoch), np.load('lossNAG32.npy'), label='NAG32')
    plt.plot(range(0, num_epoch), np.load('lossRMSprop1.npy'), label='RMSprop1')
    plt.plot(range(0, num_epoch), np.load('lossRMSprop32.npy'), label='RMSprop32')


plt.yscale("log")

plt.legend(loc='best', fontsize=12)

plt.show()