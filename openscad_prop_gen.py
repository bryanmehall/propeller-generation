import matplotlib.pyplot as plt
import numpy as np
import pylab as p
import pickle

#set hub parameters
shaft_diameter = .8
hub_diameter = 3
hub_height = 4
alpha = .1

#set propeller parameters 
radius = 25.
pitch = 8.

chord = 7.
chord_function = chord * 1

sweep = 0

x_transform = 0
y_transform = 0

bottom_camber = 0.6
bcamber_function = bottom_camber * 1
bcamber_fchord = 1/4

top_camber = 1.2
tcamber_function = top_camber * 1
tcamber_fchord = 1/7

min_thickness = .025
angle_lim = np.pi/6

resolution = 50
lenres = 100

def cs_create(frac_radius):
	chord_function = (-1.618*(frac_radius-.618)**2+1)*chord
	pitch_angle = np.arctan((pitch/(2*np.pi*radius*frac_radius)))
	theta = pitch_angle + alpha
	x_transform = (-chord_function*np.cos(theta))/2
	
#define control points
	n = 3
	
	tx1 = 0.
	ty1 = 0
	tx2 = chord_function * tcamber_fchord 
	ty2 = tcamber_function * 2
	tx3 = chord_function
	ty3 = 0
	
	bx1 = 0.
	by1 = 0.
	bx2 = chord_function * bcamber_fchord 
	by2 = bcamber_function * 2
	bx3 = chord_function
	by3 = 0.
#create control point arrays
	tcontrol_points = np.array([ (tx1, tx2, tx3) , (ty1, ty2, ty3) ])
	bcontrol_points = np.array([ (bx1, bx2, bx3) , (by1, by2, by3) ])
	
	
#transform control points	    
	
	
	transform_array = np.array( 
	[ (np.cos(theta)	, np.sin(theta)	, x_transform), 
	  (-np.sin(theta)	, np.cos(theta)	, y_transform),
	  (0				, 0		, 1          ) ] 
	)
	
	def points(array):
		rotated = np.dot(transform_array,np.vstack((array,np.array([1,1,1]))))
		Xcont = rotated[0,0:3]
		Ycont = rotated[1,0:3]
		
		def B(coorArr, i, j, t):
	   		if j == 0:
				return coorArr[i]
	   		return B(coorArr, i, j - 1, t) * (1 - t) + B(coorArr, i + 1, j - 1, t) * t
	    
		Xarr = []
		Yarr = []

		for k in range(resolution):
		    t = float(k) / (resolution-1)
		    x = B(Xcont, 0, n-1, t)
		    y = B(Ycont, 0, n-1, t)
		    Xarr.append(x)
		    Yarr.append(y)
		return Xarr,Yarr
		
	xt,yt = points(tcontrol_points)
	yt = list(np.array(yt)+min_thickness)
	xb,yb = points(bcontrol_points)

	Xarray = np.array(xt+ list(reversed(xb)))
	Yarray = np.array(yt+ list(reversed(yb)))

	return Xarray, Yarray
	
def cs_place(frac_radius):
	Xarray, Yarray = cs_create(frac_radius)
	circumfrence = 2*np.pi*radius*frac_radius
	conversion = Xarray/circumfrence
	theta = 2*np.pi*conversion
	r = radius*frac_radius
	x = tuple(r * np.cos(theta))
	y = tuple(r * np.sin(theta))
	z = tuple(Yarray)
	return x,y,z
	

xverts = []
yverts = []
zverts = []

for i in np.linspace((hub_diameter/2)/radius,1,lenres):
	x, y, z = cs_place(i)
	xverts += list(x)
	yverts += list(y)
	zverts += list(z)

#create hub
"""def hub_create():
def place_verts(side)
	Xarray, Yarray = cs_create(frac_radius)
	circumfrence = 2*np.pi*radius*frac_radius
	conversion = Xarray/circumfrence
	theta = 2*np.pi*conversion
	
x, y, z = cs_place((hub_diameter/2)/radius)
theta_start = 
theta = 2*resolution/(2*np.pi)
def cylinder(r):
	topxverts = []
	topyverts = []
	for i in range(2*resolution):
		cx = r * np.cos(theta)
		cy = r * np.sin(theta)
		topxverts += cx
		topyverts += cy"""


blade = [(float(xverts[i]), float(yverts[i]), float(zverts[i])) for i in range(0,len(xverts))]

faces = []

for j in range(0,(lenres-1)*2*resolution,2*resolution):
		for i in range(0,2*resolution-1):
			cs_fill = [(j+i,j+i+1,j+i+1+2*resolution),(j+i,j+i+1+2*resolution,j+i+2*resolution) ]
			faces += cs_fill
		faces.append((j+resolution*2-1,j,j+2*resolution))
		faces.append((j+resolution*2-1,j+2*resolution,j+4*resolution-1))
	
cap_start = lenres*2*resolution-2*resolution

for i in range(0,resolution):
	faces.append((cap_start+i,cap_start+1+i,cap_start+2*resolution-i-2))
	faces.append((cap_start+i,cap_start+2*resolution-i-2,cap_start+2*resolution-i-1))
	
for i in range(0,resolution):
	faces.append((2*resolution-i-2,1+i,i))
	faces.append((2*resolution-i-1,2*resolution-i-2,i))
points = '],['.join(
	','.join(str(y) for y in x
	) for x in blade)
	
triangles = '],['.join(
	','.join(str(y) for y in x
	) for x in faces)

openscad = """
module blade()
{{
polyhedron(
		points =[[{0}]], 
		triangles = [[{1}]]
		);
}}
difference()
{{
	union() {{
		translate ([0,0,-1.5]) cylinder (h = 4, r=1, center = true, $fn=100);
		blade();
		rotate ([0,0,180]) blade();
	
	}}
	translate ([0,0,-1.5]) cylinder (h = 5, r=0.5, center = true, $fn=100);

}}
     """.format(points,triangles)

f = open('/home/bryan/Dropbox/Shared/pcbquadcopter/prop.scad', 'w')
f.write(openscad)
	

"""data = {'coords': blade, 'faces': faces}
pickle.dump( data, open( "save1.p", "wb" ) )"""

print 'verts:',len(blade)
print 'faces:',len(faces)


	
	
