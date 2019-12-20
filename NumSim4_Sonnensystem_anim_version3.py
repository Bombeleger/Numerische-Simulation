import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as ani
import csv
plt.close()

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(r, phi)

def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)
#-------------txt-Eingabe
#pfad = "NumSim_Prakt_Sonnensystem_Teilaufgabe_1.txt"
#Gestalt der Matrix
#merkur,venus,erde,mond,mars,jupiter...
#masse *1024kg
#mean distfromsun *106km
#perihelion*106km
#aphelion*106km
#planetdata = np.loadtxt(pfad, usecols=range(10))#spalte ist die planetennummer(mond nicht vergessen)

# newton gesetz, hier venus
#f = G * m_S * planetdata[0,1]/(planetdata[1,1]**2)
#a = planetdata[3,1]/(1+f)
#
#timemax = 7
#stepcount = 1000
#dt = timemax/stepcount
#time = np.arange(0,timemax,dt)
#r = np.zeros(np.shape(time)[0])
#
#r = a*(1-f*f)/(1+f*np.cos(time[:]))
#array_xy = pol2cart(r[:], time[:])
#array_xy = np.array(array_xy)


#------------CSV-Eingabe
pfad = "planetdata20191213.csv"
planetdata1 = []

with open("planetdata20191213.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    for row in spamreader:
        planetdata1.append(row)


planetdata = np.zeros(np.shape(planetdata1))
planetdata = planetdata.astype(str)


for i in range(np.shape(planetdata1)[0]):
    planetdata[i] = np.array(planetdata1[i])
#    print(planetdata[i])

for i in range(2,np.shape(planetdata1)[0]):#ersetze alle kommata mit punkten
    for j in range(3,np.shape(planetdata1)[1]):
        planetdata[i,j] = planetdata[i,j].replace(",",".")
        planetdata[i,j] = planetdata[i,j].astype(float)#geht nicht warum auch immer

# Gravitationskonstante G
G = 6.672 * 10 ** -11

# Masse der Sonne m_S
m_S = 1988500


#newton new
r_mean = (planetdata[4,8].astype(float)+planetdata[4,7].astype(float))/2
f = G * m_S * planetdata[4,3].astype(float)/(r_mean**2)
a = planetdata[4,8].astype(float)/(1+f)

timemax = 6*np.pi
stepcount = 1000
dt = timemax/stepcount
time = np.arange(0,timemax,dt)
r = np.zeros(np.shape(time)[0])

r = a*(1-f*f)/(1+f*np.cos(time[:]))
array_xy = pol2cart(r[:], time[:])
array_xy = np.array(array_xy)

# Animation
fig = plt.figure()
ax = fig.gca()
ax.set_facecolor('black')
ax.set_xlim([- np.max(r)*1.1, np.max(r)*1.1])
ax.set_ylim([- np.max(r)*1.1, np.max(r)*1.1])
ax.plot(0,0,'o',color=[1,1,0])
lines, = ax.plot([], [], lw = 1)
points, = ax.plot([], [],  'o',lw = 1,color = planetdata[4,2])

def init():
    lines.set_data([], [])
    points.set_data([],[])
    return lines, points,

def animate(i):
    x = array_xy[ 0, :i]
    y = array_xy[ 1, :i]
    xp = x[i-1]
    yp = y[i-1]
    lines.set_data(x , y)
    points.set_data(xp, yp)
    return lines, points,

anim = ani.FuncAnimation(fig, animate, init_func=init, frames = 1000, interval = 0, blit = True, repeat = False)

plt.show()

#ax = plt.subplot(111)
#ax.set_facecolor('black')
#ax.plot(array_xy[0],array_xy[1])
#ax.plot(0,0,'o',color=[1,1,0])
