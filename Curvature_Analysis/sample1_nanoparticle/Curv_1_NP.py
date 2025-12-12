import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

start_time1 = time.time()


def plot_points(points, number=0, color='black', path=''):
    """
    Plot points in 3D space.

    Parameters:
        points (np.ndarray): Array of points to be plotted.
        number (int): Figure number (default is 0).
        color (str): Color of the points (default is 'black').
        path (str): File path to save the plot (default is an empty string).
    """
    fig = plt.figure(num=number)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, color=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title('Points on the surface of the shape')
    plt.savefig(path)
    plt.show(block=False)
    plt.close('all')


def generate_sphere_points(num_points, r=10, center=(0, 0, 0)):
    """
    Generate points uniformly distributed on the surface of a sphere.

    Parameters:
        num_points (int): Number of points on the sphere surface.
        r (float): Radius of the sphere.
        center (tuple): Coordinates of the sphere center.

    Returns:
        np.ndarray: Array of shape (num_points, 3) containing the coordinates.
    """
    points = []
    increment = math.pi * (3 - math.sqrt(5))
    offset = 2 / float(num_points)
    for i in range(num_points):
        y = i * offset - 1 + (offset / 2)
        rad = math.sqrt(1 - y * y)
        phi = i * increment
        x = r * math.cos(phi) * rad + center[0]
        z = r * math.sin(phi) * rad + center[2]
        points.append((x, r * y + center[1], z))
    return np.array(points)


def save_text(inputs, header, fmt='%11.3f', path='test.txt'):
    """
    Save data to a text file.

    Parameters:
        inputs (np.ndarray): Data to be saved.
        header (str): Header row written at the beginning of the file.
        fmt (str): Format string for each element (default is '%11.3f').
        path (str): File path to save the data.
    """
    with open(path, 'w') as file:
        file.write(header)
        np.savetxt(file, inputs, fmt=fmt, delimiter='\t\t')


def z_predict(x, y, p, q, r_coef, s, t, c):
    """
    Predict the value of z using a quadratic form:
    z = p*(x^2) + q*(y^2) + r*(x*y) + s*x + t*y + c

    Parameters:
        x (float or np.ndarray): x-coordinate(s).
        y (float or np.ndarray): y-coordinate(s).
        p, q, r_coef, s, t, c (float): Coefficients of the quadratic form.

    Returns:
        float or np.ndarray: Predicted z value(s).
    """
    return p * (x ** 2) + q * (y ** 2) + r_coef * (x * y) + s * x + t * y + c


def quadratic_fitting(points):
    """
    Fit the points in local coordinates by a quadratic form:
    z = p*(x^2) + q*(y^2) + r*(x*y) + s*x + t*y + c

    Parameters:
        points (np.ndarray): Array of shape (n, 3).

    Returns:
        tuple: (p, q, r, s, t, c, r_squared, check_determinant)
    """
    n = points.shape[0]
    A = np.zeros((n, 6))
    A[:, 0] = points[:, 0] ** 2
    A[:, 1] = points[:, 1] ** 2
    A[:, 2] = points[:, 0] * points[:, 1]
    A[:, 3] = points[:, 0]
    A[:, 4] = points[:, 1]
    A[:, 5] = 1
    B = points[:, 2].reshape(-1, 1)
    A_transpose = A.T

    if np.linalg.det(np.dot(A_transpose, A)) != 0:
        check_determinant = True
        X = np.dot(np.linalg.inv(np.dot(A_transpose, A)), np.dot(A_transpose, B))
        p, q, r_coef, s, t, c = X.flatten()
        z_fit = z_predict(points[:, 0], points[:, 1], p, q, r_coef, s, t, c)
        sse = np.sum((points[:, 2] - z_fit) ** 2)
        sst = np.sum((points[:, 2] - np.mean(points[:, 2])) ** 2)
        r_squared = 1 - (sse / sst)
    else:
        check_determinant = False
        p, q, r_coef, s, t, c, r_squared = (0, 0, 0, 0, 0, 0, 0)
    return p, q, r_coef, s, t, c, r_squared, check_determinant


def euclidean_distance(point_i, point_j):
    """
    Calculate the Euclidean distance between two 3D points.

    Parameters:
        point_i (list or np.ndarray): [xi, yi, zi] of the first point.
        point_j (list or np.ndarray): [xj, yj, zj] of the second point.

    Returns:
        float: Euclidean distance.
    """
    return math.sqrt(
        (point_i[0] - point_j[0]) ** 2 +
        (point_i[1] - point_j[1]) ** 2 +
        (point_i[2] - point_j[2]) ** 2
    )


def inertia_matrix(points):
    """
    Calculate the inertia matrix for a set of points.

    Parameters:
        points (np.ndarray): Array of shape (n, 3).

    Returns:
        tuple: (center of mass, centered points, inertia matrix)
    """
    n = points.shape[0]
    com = np.mean(points, axis=0)
    points_centered = points - com
    I = np.zeros((3, 3))
    for point in points_centered:
        I[0, 0] += point[1] ** 2 + point[2] ** 2
        I[1, 1] += point[0] ** 2 + point[2] ** 2
        I[2, 2] += point[0] ** 2 + point[1] ** 2
        I[0, 1] -= point[0] * point[1]
        I[0, 2] -= point[0] * point[2]
        I[1, 2] -= point[1] * point[2]

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return com, points_centered, I


def cal_curv_param(p, q, r_coef, s, t, c):
    """
    Calculate principal curvatures and Gaussian and mean curvatures.

    Parameters:
        p, q, r_coef, s, t, c (float): Coefficients of the quadratic form.

    Returns:
        tuple: (c1_curvature, c2_curvature, KG, Km)
    """
    E = 1 + s ** 2
    F = s * t
    G = 1 + t ** 2
    L = 2 * p
    M = r_coef
    N = 2 * q

    A1 = np.linalg.inv(np.array([[E, F], [F, G]]))
    A2 = np.array([[L, M], [M, N]])
    A_curv = np.dot(A1, A2)
    eigen_values, _ = np.linalg.eig(A_curv)
    c1_curv, c2_curv = eigen_values
    KG = c1_curv * c2_curv
    Km = abs(0.5 * (c1_curv + c2_curv))
    if c1_curv * c2_curv < 0:
        Km = -Km
    return c1_curv, c2_curv, KG, Km


def cal_triangle_area(p1, p2, p3):
    """
    Calculate the area of a triangle defined by three points in 3D space.

    Parameters:
    - p1 (list): List containing the coordinates [x1, y1, z1] of the first point.
    - p2 (list): List containing the coordinates [x2, y2, z2] of the second point.
    - p3 (list): List containing the coordinates [x3, y3, z3] of the third point.

    Returns:
    - area (float): Area of the triangle.
    """
    side_a = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    side_b = math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2 + (p1[2] - p3[2]) ** 2)
    side_c = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2 + (p2[2] - p3[2]) ** 2)

    # Calculate the semi-perimeter of the triangle
    semiperimeter = (side_a + side_b + side_c) / 2

    # Calculate the area using Heron's formula
    area = math.sqrt(semiperimeter * (semiperimeter - side_a) * (semiperimeter - side_b) * (semiperimeter - side_c))

    return area


def patch_area(points):
    """
    Calculate the surface area of a patch using Delaunay triangulation.

    Parameters:
        points (np.ndarray): Array of shape (n, 3).

    Returns:
        float: Surface area of the patch.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    points_2d = np.vstack([x, y]).T
    tri = Delaunay(points_2d)

    # Calculation of patch area
    patch_area = 0
    for triangle in tri.simplices:
        pt1 = points[triangle[0]]
        pt2 = points[triangle[1]]
        pt3 = points[triangle[2]]
        patch_area += cal_triangle_area(pt1, pt2, pt3)
    return patch_area



def curvature(points, patch_distance):
    """
    - points (numpy.ndarray: n*3): Array containing the coordinates of points on the surface of a patch.
                                   Each row represents a point.
    - patch_distance (float): Distance used for patch estimation.

    Returns:
    - curvature_values (numpy.ndarray: n*7): Array containing the curvature values corresponding to each
                                             point on the surface.
      n: number of points on the surface.
      It includes 7 columns for each point:
      1. Number of each point (integer)
      2. c1: First principal curvature (float)
      3. c2: Second principal curvature (float)
      4. KG: Gaussian curvature (float)
      5. Km: Mean curvature (float)
      6. R-squared: R-squared for fitting quadratic forms to surface of a patch (float)
      7. Surface area: Surface area of each patch (float)
    """
    n = (np.shape(points))[0]

    # Define a list of patches
    patch_list = []
    for i in range(n):
        patch_list.append([i])

    # Finding the list of a patch for each point
    for i in range(n):
        for j in range(n):
            if ((euclidean_distance(points[i], points[j])) < patch_distance) and (i != j):
                patch_list[i].append(j)

    # Calculating the curvature parameters
    curve_params_array = np.zeros((n, 7))
    for i in range(n):
        points_patch_array = np.array(points[(np.array(patch_list[i]))])
        if (np.shape(points_patch_array)[0]) >= 6:
            # 6 is a limitation based on the quadratic form of the plan, which typically has 6 parameters
            com, points_patch_array_centered, I = inertia_matrix(points_patch_array)
            eigenValues, eigenVectors = np.linalg.eig(I)
            eigenValues_idx = (abs(eigenValues)).argsort()[::1]
            #eigenValues = eigenValues[eigenValues_idx]
            eigenVectors = eigenVectors[:, eigenValues_idx]
            principal_inertia_axes = eigenVectors / np.linalg.norm(eigenVectors, axis=0, keepdims=True)

            # Transforming the points to the local axis
            points_patch_array_centered_local = np.matmul(points_patch_array_centered, principal_inertia_axes)

            # Fitting the quadratic form of the plane
            P, Q, R, S, T, C, R_squared, check_determinant = quadratic_fitting(points_patch_array_centered_local)

            if (check_determinant == True) and (R_squared >= R_squared_limit):
                surface_area_patch = patch_area(points_patch_array_centered_local)
                c1_curvature, c2_curvature, KG, Km = cal_curv_param(P, Q, R, S, T, C)
                curve_params_array[i] = [i + 1, c1_curvature, c2_curvature, KG, Km, R_squared, surface_area_patch]
            else:
                # If the determinant of the fitting matrix is 0 or the R-squared is less than the defined limit, return "None"  for curvature parameters
                curve_params_array[i] = [i + 1, None, None, None, None, None, None]
        else:
            # If the number of points in the patch is less than 6, return "None" for curvature parameters
            curve_params_array[i] = [i + 1, None, None, None, None, None, None]
    return curve_params_array


def plot_color_bar(points, kg, km,
                   path_kg='KG_color_bar.jpeg',
                   path_km='Km_color_bar.jpeg'):
    """
    Plot 3D scatter plots with color mapping for KG and Km.

    Parameters:
        points (np.ndarray): Array of shape (n, 3) with coordinates.
        kg (np.ndarray): Gaussian curvature values.
        km (np.ndarray): Mean curvature values.
        path_kg (str): File path for KG plot.
        path_km (str): File path for Km plot.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Plot KG
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    kg_min, kg_max = -0.15, 0.15  # it can be changed based on the range of KG
    sc = ax.scatter(x, y, z, c=kg, cmap='jet', vmin=kg_min, vmax=kg_max, s=30)
    plt.colorbar(sc)
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.savefig(path_kg)
    plt.show(block=False)
    plt.close('all')

    # Plot Km
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    km_min, km_max = -0.5, 0.5  # it can be changed based on the range of Km
    sc = ax.scatter(x, y, z, c=km, cmap='jet', vmin=km_min, vmax=km_max, s=30)
    plt.colorbar(sc)
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.savefig(path_km)
    plt.show(block=False)
    plt.close('all')


# Main code
R_squared_limit = 0.9  # Limitation of R-squared for fitting quadratic forms to surface of a patch
patch_distance = 0.5  # unit: nm

# Saving the file and diagrams
# Plotting the points
# get points from file
input_file = 'sample1.txt'
points_list = []
with open(input_file, 'r') as file1:
    lines = file1.readlines()
    for line in lines:
        x, y, z = line.split()
        points_list.append((float(x), float(y), float(z)))

points = np.array(points_list)  # Insert the points of the surface you want to analyze
num_points = points.shape[0]
print('No. of points:', num_points, '\t', 'patch_distance:', patch_distance)

plot_points(points, number=0, color='black', path='points.jpeg')

curve_params_array = curvature(points, patch_distance)
curve_params_array[:, 0] = curve_params_array[:, 0].astype(int)

# Saving the coordinate of the points and curvature parameters as a text file
save_text(
    points,
    header="\t\tX\t\t\t\tY\t\t\t\tZ\n",
    path='points.txt'
)
fmt_list = ['%9d' if i == 0 else '%11.3f' for i in range(curve_params_array.shape[1])]
save_text(
    curve_params_array,
    header=("\tbead No\t\t\t\tc1\t\t\t\tc2\t\t\t\tKG\t\t\t\tKm\t\t\t\tR^2\t\t\tpatch area\n"),
    fmt=fmt_list,
    path='curvature_param.txt'
)

# KG histogram (bar format)
hist_kg, bins_kg, _ = plt.hist(
    curve_params_array[:, 3],
    bins=100,
    color='green',
    label='KG',
    range=(-0.3, 0.3),
    density=True
)
plt.title("KG")
plt.ylabel("Normalized Frequency")
plt.savefig("KG_hist_bar.jpeg")
plt.show(block=False)
plt.close('all')

bins_centers_kg = 0.5 * (bins_kg[1:] + bins_kg[:-1])
kg_hist_line_array = np.column_stack((bins_centers_kg, hist_kg))
save_text(
    kg_hist_line_array,
    header="\tbins_centers\tKG_frequency\n",
    path="KG_hist.txt"
)
plt.plot(bins_centers_kg, hist_kg, "-o")
plt.title("KG")
plt.xlabel("Bin Centers")
plt.ylabel("Normalized Frequency")
plt.savefig("KG_hist_line.jpeg")
plt.show(block=False)
plt.close('all')

# Km histogram (bar format)
hist_km, bins_km, _ = plt.hist(
    curve_params_array[:, 4],
    bins=100,
    color="cyan",
    label="Km",
    range=(-0.8, 0.8),
    density=True
)
plt.title("Km")
plt.ylabel("Normalized Frequency")
plt.savefig("Km_hist_bar.jpeg")
plt.show(block=False)
plt.close('all')

bins_centers_km = 0.5 * (bins_km[1:] + bins_km[:-1])
km_hist_line_array = np.column_stack((bins_centers_km, hist_km))
save_text(
    km_hist_line_array,
    header="\tbins_centers\tKm_frequency\n",
    path="Km_hist.txt"
)
plt.plot(bins_centers_km, hist_km, "-o")
plt.title("Km")
plt.xlabel("Bin Centers")
plt.ylabel("Normalized Frequency")
plt.savefig("Km_hist_line.jpeg")
plt.show(block=False)
plt.close('all')

# R^2 scatter plot
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(curve_params_array[:, 0], curve_params_array[:, 5],
           color="black", s=2)
plt.savefig("R_squared.jpeg")
plt.show(block=False)
plt.close('all')

plot_color_bar(points, kg=curve_params_array[:, 3],
               km=curve_params_array[:, 4])

end_time1 = time.time()
print("Elapsed time:", round((end_time1 - start_time1) / 60), "mins")
