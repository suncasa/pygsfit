import sunpy.map
from astropy.coordinates import SkyCoord
import astropy.units as u
def submap_of_file1(file1, map2):
    """
    Modify the WCS of file2 to match the FOV of file1, while retaining its spatial resolution.

    Parameters:
    file1 (str): Path to the first FITS file.
    file2 (str): Path to the second FITS file.

    Returns:
    sunpy.map.GenericMap: A SunPy map with the modified WCS.
    """
    # Load the FITS files as SunPy maps
    map1 = sunpy.map.Map(file1)
    #map2 = sunpy.map.Map(file2)

    # Get the FOV of the second map
    x_range = u.Quantity([map2.bottom_left_coord.Tx, map2.top_right_coord.Tx])
    y_range = u.Quantity([map2.bottom_left_coord.Ty, map2.top_right_coord.Ty])

    # Create ao submap of the first map using the FOV of the second map
    bl = SkyCoord(x_range[0], y_range[0], frame=map1.coordinate_frame)
    tr = SkyCoord(x_range[1], y_range[1], frame=map1.coordinate_frame)
    submap1 = map1.submap(bl,  top_right = tr)

    return submap1

def resize_array(array, new_shape):
    from scipy.ndimage import zoom
    zoom_factor = (new_shape[0] / array.shape[0], new_shape[1] / array.shape[1])
    resized_array = zoom(array, zoom_factor, order=3)  # order=3 for cubic interpolation
    return resized_array