import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pylab as p
#import matplotlib.axes3d as p3
import mpl_toolkits.mplot3d.axes3d as p3
fig=p.figure()
ax = p3.Axes3D(fig)
matplotlib.rc('text', usetex=True)
#set prop parameters
radius = 25.
pitch = 8.
chord = 10.
bcamber = -0.6
tcamber = 1.
min_thickness = .025
resolution = 25
	
	

def cs_place(frac_radius):
	Xarr, Yarr = cs_create(frac_radius)
	
	circumfrence = 2*np.pi*radius*frac_radius
	conversion = Xarr/circumfrence
	theta = 2*np.pi*conversion
	r = radius*frac_radius
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	z = Yarr
	return x,y,z
	



def cs_create(frac_radius):
	#define control points
	n = 3
	bx1 = 0.
	by1 = 0.
	bx2 = chord / 4
	by2 = bcamber * 2
	bx3 = chord
	by3 = 0.

	tx1 = 0.
	ty1 = min_thickness
	tx2 = chord / 7
	ty2 = tcamber * 2
	tx3 = chord
	ty3 = min_thickness

	bXconti = np.array( [bx1, bx2, bx3] )
	bYconti = np.array( [by1, by2, by3] )

	tXconti = np.array( [tx1, tx2, tx3] )
	tYconti = np.array( [ty1, ty2, ty3] )

	#adjust control points for pitch.

	def pitch_angle(frac_radius):
	    return np.arctan((pitch/(2.*np.pi*radius*frac_radius)))

	def rotate(frac_radius,arrX,arrY):
		theta = pitch_angle(frac_radius)
		transform_arr = np.array( [ (np.cos(theta), np.sin(theta)), (-np.sin(theta),np.cos(theta)) ] )
		cont = np.vstack((arrX,arrY))
		trans = np.dot(transform_arr,cont)
		return trans

	bcont = rotate(frac_radius,bXconti,bYconti)
	bXcont = bcont[0,0:3]
	bYcont = bcont[1,0:3]

	tcont = rotate(frac_radius,tXconti,tYconti)
	tXcont = tcont[0,0:3]
	tYcont = tcont[1,0:3]
	#calculate points
	def B(coorArr, i, j, t):
	    if j == 0:
		  return coorArr[i]
	    return B(coorArr, i, j - 1, t) * (1 - t) + B(coorArr, i + 1, j - 1, t) * t
	    
	tXarr = []
	tYarr = []

	bXarr = []
	bYarr = []

	#bottom points
	for k in range(resolution):
	    t = float(k) / (resolution-1)
	    x = B(bXcont, 0, n-1, t)
	    y = B(bYcont, 0, n-1, t)
	    bXarr.append(x)
	    bYarr.append(y)

	#top points
	for l in range(resolution):
	    t = float(l) / (resolution-1)
	    x = B(tXcont, 0, n-1, t)
	    y = B(tYcont, 0, n-1, t)
	    tXarr.append(x)
	    tYarr.append(y)

	Xarr = np.array(bXarr + list(reversed(tXarr)) + [bXarr[0]])
	Yarr = np.array(bYarr + list(reversed(tYarr)) + [bYarr[0]])

	#plot points
	"""plt.plot(Xarr, Yarr)
	#plt.plot(tXarr, tYarr)
	plt.xlabel('x coordinates (mm)')
	plt.ylabel('y coordinates (mm)')
	plt.text(1,8,r'chord:10 mm \\ camber:1.2 mm \\ Resolution:100' )
	plt.axis([0, 10,-5 , 5])
	plt.show()"""

	return Xarr, Yarr
	

xverts = []
yverts = []
zverts = []
for i in np.arange(.1,1.,.05):
	x, y, z = cs_place(i)
	print x,y,z
	xverts += list(x)
	yverts += list(y)
	zverts += list(z)
	
print xverts
print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
print yverts
print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
print zverts
	
ax.plot(xverts, yverts, zverts)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-25,25)
ax.set_ylim3d(-25,25)
ax.set_zlim3d(-25,25)

p.show()
	
	
