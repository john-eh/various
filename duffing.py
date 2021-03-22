import numpy as np
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.pyplot as plt

def duffing_eq(y,t, gamma, delta, omega):
        
    x, dx = y
    ddx = -delta*dx + x - (x**3) + gamma*np.cos(omega*t)
    return dx, ddx

def solve_duffing_eq(y0,gamma,delta,omega,max_t,start_t,period_its):
   
    period = 2*np.pi/omega
    dt = period / period_its
    t = np.linspace(0,max_t,num = int(max_t/dt))
    
    y = odeint(duffing_eq,y0,t, args=(gamma, delta, omega))
    
    return y[int(start_t/dt):].T


# initial conditions and parameters
y0 = [0.52,0]
gamma,delta, omega = 0.4, 0.1, 1.4

# time points
max_t = 30000
start_t = 300
period_its = 150

x, dx = solve_duffing_eq(y0,gamma,delta,omega,max_t,start_t,period_its)

fig = plt.figure(figsize=(16,12))
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
line = ax.scatter([], [], s=4, lw=0)

def update(i):
    x_draw = x[i::period_its]
    y_draw = dx[i::period_its]
    X = np.vstack((x_draw, y_draw))
    line.set_offsets(X.T)
    return line

anim = animation.FuncAnimation(fig, update,frames=period_its, interval=20)

anim.save('duffing_animation.mp4', fps=15, bitrate = 8000, extra_args=['-vcodec', 'libx264'])

plt.show()
