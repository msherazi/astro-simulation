import numpy as np

rightAscensions = []

aldebaranRA = [4, 35, 55]
rigelRA = [5, 14, 32]
betelgeuseRA = [5, 55, 10]
siriusRA = [6, 45, 9]
procyonRA = [7, 39, 18]

rightAscensions.append(aldebaranRA)
rightAscensions.append(rigelRA)
rightAscensions.append(betelgeuseRA)
rightAscensions.append(siriusRA)
rightAscensions.append(procyonRA)

declinations = []

aldebarandec = [16, 30, 33]
rigeldec = [-8, -12, -6]
betelgeusedec = [7,24,25]
siriusdec = [-16, -42, -58]
procyondec = [5, 13, 30]

declinations.append(aldebarandec)
declinations.append(rigeldec)
declinations.append(betelgeusedec)
declinations.append(siriusdec)
declinations.append(procyondec)

radeg = []
decdeg = []

starNames = ["Aldebaran", "Rigel", "Betelgeuse", "Sirius", "Procyon"]

coordinates = []

for i in range(0, 5):
    currentRA = rightAscensions[i]
    currentRA[0] = currentRA[0] * 15.0 
    currentRA[1] = currentRA[1]/4.0
    currentRA[2] = currentRA[2]/240.0
    RA_sum = sum(currentRA)
    rarad = np.deg2rad(RA_sum)
    radeg.append(rarad)

    currentdec = declinations[i]
    currentdec[1] = currentdec[1]/60.0
    currentdec[2] = currentdec[2]/3600.0
    dec_sum = sum(currentdec)
    decrad = np.deg2rad(dec_sum)
    decdeg.append(decrad)

    coordinate = [1, decrad, rarad]
    coordinates.append(coordinate)



coordinatesnp = np.array(coordinates)

for i in range(0, 5):
    for j in range(0, 5):
        if(i != j):
            theta1 = 0.5*np.pi-coordinatesnp[i][1]
            phi1 = coordinatesnp[i][2]

            theta2 = 0.5*np.pi-coordinatesnp[j][1]
            phi2 = coordinatesnp[j][2]

            star1 = np.array([np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)])
            star2 = np.array([np.sin(theta2)*np.cos(phi2), np.sin(theta2)*np.sin(phi2), np.cos(theta2)])

            angularSep = np.arccos(np.dot(star1, star2))

            print("Angular separation between " + str(starNames[i]) + " and " + str(starNames[j]) + ":" + str(angularSep * (180/np.pi)))