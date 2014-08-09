# Copyright (C) 2013  University of Saskatchewan SuperDARN group
# Full license can be found in LICENSE.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
.. module:: coordUtils
   :synopsis: A module for manipulating coordinates

.. moduleauthor:: mrwessel

****************************
**Module**: utils.coordUtils
****************************
**Functions**:
  * :func:`utils.coordUtils.coordConv`: Convert between geographic, 
        AACGM, and MLT coordinates
  * :func:`utils.coordUtils.planeRot`: Rotate coordinates in the plane

Written by Matt W. based on code by...Sebastien?
"""

def coordConv(lon, lat, altitude, start, end, dateTime=None):
    """Convert between geographical, AACGM, and MLT coordinates.  
        dateTime must be set to use any AACGM systems.
  
    **Args**: 
        * **lon**: longitude (MLT must be in degrees, not hours)
        * **lat**: latitude
        * **altitude**: altitude to be used (km).
        * **start**: coordinate system of input. Options: 'geo', 'mag',
            'mlt'
        * **end**: desired output coordinate system. Options: 'geo', 
            'mag', 'mlt'
        * **[dateTime]**: python datetime object. Default: None
    **Returns**:
        * **lon,lat**: lists
    **Example**:
        ::
    
        lon, lat = coordConv(lon, lat, 'geo', 'mlt', 
                             dateTime=datetime(2012,3,12,0,56))
        
    written by Matt W., 2013-09 based on code by...Sebastien?
    """
    import numpy as np
    
    from models import aacgm

    date_time = dateTime

    # Define acceptable coordinate systems.
    coords_dict = {"mag": "AACGM",
                   "geo": "Geographic",
                   "mlt": "MLT"}
    
    # Check that the coordinates are possible.
    if end and end not in coords_dict:
        print "Invalid end coordinate system given ({}):"
        print "setting 'mag'".format(end)
        end = "mag"
    if start and start not in coords_dict:
        print "Invalid start coordinate system given ({}):"
        print "setting '{}'".format(start,end)
        start = end 
    
    # MLT requires a datetime.
    if start == "mlt" or end == "mlt": 
        assert(date_time is not None),\
                "date_time must be provided for MLT coordinates to work."
    # Mag requires a datetime.
    if start == 'mag' or end == 'mag': 
        assert(date_time is not None),\
                "date_time must be provided for MAG coordinates to work."

    # Sanitise inputs.
    if isinstance(lon, int):
        lon = float(lon)
        lat = float(lat)
    is_float = isinstance(lon, float)
    is_list = isinstance(lon, (list, tuple))
    if not (is_float or is_list):
        assert(isinstance(lon, np.ndarray)),\
                "Must input int, float, list, or numpy array."

    # Make the inputs into numpy arrays because single element lists 
    # have no len.
    lon = np.array(lon)
    lat = np.array(lat)

    # Test whether we are using the same altitude for everything.
    if np.size(alt) == 1:
        altitude = [altitude]*np.size(lon)

    # If there is an actual conversion to do:
    if start and end and start != end:
        # Set the conversion specifier.
        trans = start+'-'+end

        # Geographical and AACGM conversions:
        if trans in ['geo-mag','mag-geo']:
            flag = 0 if trans == 'geo-mag' else 1
            lont = lon
            latt = lat
            nlon, nlat = np.size(lont), np.size(latt)
            shape = lont.shape
            lat, lon, _ = aacgm.aacgmConvArr(list(latt.flatten()), 
                                             list(lont.flatten()), altitude,
                                             date_time.year,flag)
            lon = np.array(lon).reshape(shape)
            lat = np.array(lat).reshape(shape)

        # Geographical and MLT conversions:
        elif trans in ['geo-mlt','mlt-geo']:
            flag = 0 if trans == 'geo-mlt' else 1
            if flag:
                latt = lat
                lon_mlt = lon
                shape = lon_mlt.shape
                nlon, nlat = np.size(lon_mlt), np.size(latt)
                lon_mag = []
                # Find MLT at zero mag lon.
                mlt_0 = aacgm.mltFromYmdhms(date_time.year,date_time.month,
                                            date_time.day, date_time.hour,
                                            date_time.minute, date_time.second,
                                            0.)
                for el in range(nlon):
                    # Convert input mlt from degrees to hours.
                    mlt = lon_mlt.flatten()[el]*24./360.%24.
                    # Calculate mag lon.
                    mag = (mlt - mlt_0)*15.%360.
                    # Put in -180 to 180 range.
                    if mag > 180.: mag -= 360.
                    # Stick on end of the list.
                    lon_mag.append(mag)
                # Make into a numpy array.
                lont = np.array(lon_mag).reshape(shape)
                # Convert mag to geo.
                lat, lon, _ = aacgm.aacgmConvArr(list(latt.flatten()), 
                                                 list(lont.flatten()), 
                                                 altitude, date_time.year, 
                                                 flag)
                # Make into numpy arrays.
                lon = np.array(lon).reshape(shape)
                lat = np.array(lat).reshape(shape)
            else:
                # MLT list.
                lon_mlt = []
                lont = lon
                latt = lat
                nlon, nlat = np.size(lont), np.size(latt)
                shape = lont.shape
                # Convert geo to mag.
                lat, lon, _ = aacgm.aacgmConvArr(list(latt.flatten()), 
                                                 list(lont.flatten()), 
                                                 altitude, date_time.year, 
                                                 flag)
                for lonel in range(nlon):
                    # Get MLT from mag lon.
                    mlt = aacgm.mltFromYmdhms(date_time.year,date_time.month,
                                              date_time.day, date_time.hour,
                                              date_time.minute,
                                              date_time.second, lon[lonel]) 
                    # Convert hours to degrees.
                    lon_mlt.append(mlt/24.*360.)
                # Make into numpy arrays.
                lon = np.array(lon_mlt).reshape(shape)
                lat = np.array(lat).reshape(shape)

        # Magnetic and MLT conversions:
        elif trans in ['mag-mlt','mlt-mag']:
            flag = 0 if trans == 'mag-mlt' else 1
            if flag:
                lon_mag=[]
                lon_mlt = lon
                nlon = np.size(lon_mlt)
                shape = lon_mlt.shape
                # Find MLT of zero mag lon.
                mlt_0 = aacgm.mltFromYmdhms(date_time.year, date_time.month,
                                            date_time.day, date_time.hour,
                                            date_time.minute, date_time.second,
                                            0.)     
                for el in range(nlon):
                    # Convert MLT from degrees to hours.
                    mlt = lon_mlt.flatten()[el]*24./360.%24.
                    # Calculate mag lon from MLT.
                    mag = (mlt - mlt_0)*15.%360.
                    # Put in -180 to 180 range.
                    if mag > 180.: mag -= 360.
                    lon_mag.append(mag)
                lon = np.array(lon_mag).reshape(shape)
                lat = lat.reshape(shape)
            else:
                lon_mlt = []
                lont = lon
                nlon = np.size(lont)
                shape=lont.shape
                for lonel in range(nlon):
                    # Find MLT from mag lon and datetime.
                    mlt = aacgm.mltFromYmdhms(date_time.year, date_time.month,
                            date_time.day, date_time.hour, date_time.minute,
                            date_time.second, lont.flatten()[lonel])
                    # Convert hours to degrees.
                    lon_mlt.append(mlt/24.*360.)
                lon = np.array(lon_mlt).reshape(shape)
                lat = lat.reshape(shape)
        
    # Convert outputs to input type.
    if is_list:
        lon = list(lon.flatten())
        lat = list(lat.flatten())
    elif is_float:
        lon = list(lon.flatten())[0]
        lat = list(lat.flatten())[0]
    # Otherwise it stays a numpy array.

    return lon, lat


def planeRot(x, y, theta):
    """Rotate coordinates in the plane.
  
    **Args**: 
        * **x**: x coordinate
        * **y**: y coordinate
        * **theta**: angle of rotation of new coordinate frame
    **Returns**:
        * **x_prime,y_prime**: x, y coordinates in rotated frame
    **Example**:
        ::
        import numpy as np
        x, y = planeRot(x, y, 30.*np.pi/180.)
        
    written by Matt W., 2013-09
    """
    import numpy as np

    oldx, oldy = np.array(x), np.array(y)

    x = oldx*np.cos(theta) + oldy*np.sin(theta)
    y = -oldx*np.sin(theta) + oldy*np.cos(theta)

    return x, y


# Some testing stuff.
if __name__ == "__main__":
    from datetime import datetime
    import numpy

    print
    print "All of these results may have varying sigfigs."
    print "The expected values were found on a 32-bit system."
    print
    print "Single coord pair tests"
    print
    print "Test of list -> list"
    print "Expected:  ([50.700000000000003], [34.5])"
    print "Result:    " + str(coordConv([50.7],[34.5],300.,'geo','geo'))
    print
    print "Test of float -> float"
    print "Expected:  (50.700000000000003, 34.5)"
    print "Result:    " + str(coordConv(50.7,34.5,300.,'geo','geo'))
    print
    print "Test of int -> float"
    print "Expected:  (50.0, 34.0)"
    print "Result:    " + str(coordConv(50,34,300.,'geo','geo'))
    print
    print "Test of numpy array -> numpy array"
    print "Expected:  (array([ 50.7]), array([ 34.5]))"
    print "Result:    " + str(coordConv(numpy.array([50.7]),numpy.array([34.5]),300.,'geo','geo'))
    print
    print "geo to geo, mag to mag, mlt to mlt, ashes to ashes"
    print "Expected:  (50.700000000000003, 34.5)"
    print "Result:    " + str(coordConv(50.7,34.5,300.,'geo','geo'))
    print "Result:    " + str(coordConv(50.7,34.5,300.,'mag','mag',dateTime=datetime(2013,7,23,12,6,34)))
    print "Result:    " + str(coordConv(50.7,34.5,300.,'mlt','mlt',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "geo to mag"
    print "Expected: (123.71642616363432, 31.582924632749929)"
    print "Result:   " + str(coordConv(50.7,34.5,300.,'geo','mag',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "geo to mlt"
    print "Expected: (229.34000961374397, 31.582924632749929)"
    print "Result:   " + str(coordConv(50.7,34.5,300.,'geo','mlt',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mag to geo"
    print "Expected: (50.563320914903102, 32.408924471374895)"
    print "Result:   " + str(coordConv(123.53805352405843,29.419420613372086,300.,'mag','geo',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mlt to geo"
    print "Expected: (50.563320914903137, 32.408924471374895)"
    print "Result:   " + str(coordConv(229.16163697416806,29.419420613372086,300.,'mlt','geo',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mag to mlt"
    print "Expected: (229.16163697416806, 29.419420613372086)"
    print "Result:   " + str(coordConv(123.53805352405843,29.419420613372086,300.,'mag','mlt',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mlt to mag"
    print "Expected: (123.53805352405843, 29.419420613372086)"
    print "Result:   " + str(coordConv(229.16163697416806,29.419420613372086,300.,'mlt','mag',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "Coord array tests"
    print
    print "geo to geo, mag to mag, mlt to mlt"
    print "Expected: ([50.700000000000003, 53.799999999999997], [34.5, 40.200000000000003])"
    print "Result    " + str(coordConv([50.7,53.8],[34.5,40.2],300.,'geo','geo'))
    print "Result    " + str(coordConv([50.7,53.8],[34.5,40.2],300.,'mag','mag',dateTime=datetime(2013,7,23,12,6,34)))
    print "Result    " + str(coordConv([50.7,53.8],[34.5,40.2],300.,'mlt','mlt',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "geo to mag"
    print "Expected: ([123.65800072339718, 126.97463420949806], [30.892913121194589, 37.369211032553089])"
    print "Result    " + str(coordConv([50.7,53.8],[34.5,40.2],[200.,300.],'geo','mag',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "geo to mlt"
    print "Expected: ([229.28158417350679, 232.59821765960771], [30.892913121194589, 37.369211032553089])"
    print "Result    " + str(coordConv([50.7,53.8],[34.5,40.2],[200.,300.],'geo','mlt',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mag to geo"
    print "Expected: ([50.615511474515607, 53.648287906901672], [33.150771171950133, 38.637420715148586])"
    print "Result    " + str(coordConv([123.53805352405843, 126.76454464467615],[29.419420613372086, 35.725172012254788],[200.,300.],'mag','geo',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mlt to geo"
    print "Expected: ([50.563320914903137, 53.648287906901672], [32.408924471374895, 38.637420715148586])"
    print "Result    " + str(coordConv([229.16163697416806, 232.38812809478577], [29.419420613372086, 35.725172012254788],300.,'mlt','geo',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mag to mlt"
    print "Expected: ([229.16163697416806, 232.38812809478577], [29.419420613372086, 35.725172012254788])"
    print "Result    " + str(coordConv([123.53805352405843, 126.76454464467615],[29.419420613372086, 35.725172012254788],300.,'mag','mlt',dateTime=datetime(2013,7,23,12,6,34)))
    print
    print "mlt to mag"
    print "Expected: ([123.53805352405843, 126.76454464467615], [29.419420613372086, 35.725172012254788])"
    print "Result    " + str(coordConv([229.16163697416806, 232.38812809478577], [29.419420613372086, 35.725172012254788],200.,'mlt','mag',dateTime=datetime(2013,7,23,12,6,34)))
    print
