# Juan Pablo Zuluaga C
# Sergio Hernandez
# Proc img y video 2021-3

import cv2
import sys
import os
import numpy as np

points = []
flag = False


# Funcion para ver el evento del click derecho y el click izquierdo

def click(event, x, y, flags, param):
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        flag = True


# Funcion para leer las imagenes del path dado en los argumentos
def img_lect():
    cont = 1
    path = sys.argv[1]

    # numero de imagenes que contiene el path
    N = len(os.listdir(path))
    print("Se recibieron " + str(N) + " imagenes ")
    images = []

    # Se recorre el path y se guardan las imagenes
    for i in range(N):
        image_name = "image_" + str(i + 1) + ".jpeg"
        path_file = os.path.join(path, image_name)
        images.append(cv2.resize(cv2.imread(path_file), (640, 480), interpolation=cv2.INTER_AREA))
    return N, images


# Funcion para realizar la homografia
def homografia(img, img_referencia):
    global points, flag
    # Se concatenan las imagenes de entrada
    image_concat = cv2.hconcat([img, img_referencia])
    image_draw = image_concat.copy()

    # Se definen las listas de puntos
    points1 = []
    points2 = []

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)

    state = True  # Rojo
    while True:
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
        # Sale del while si se oprime x
        if key == ord("x"):
            break
        if flag:
            # Se usa la flag para intercalar los colores y la lista de puntos en donde se guarda
            flag = False
            # Se guardan los puntos
            if (state):
                if (len(points2) > 0):
                    points2.pop(-1)
                    state = not state
            else:
                if (len(points1) > 0):
                    points1.pop(-1)
                    state = not state
            image_draw = image_concat.copy()
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [0, 0, 255], -1) for punto in points1]
            [cv2.circle(image_draw, (punto[0] + img.shape[1], punto[1]), 3, [255, 0, 0], -1) for punto in points2]

        if len(points) > 0:
            if (state):
                state = False
                points1.append((points[0][0], points[0][1]))
                points = []
            else:
                state = True
                points2.append((points[0][0] - img.shape[1], points[0][1]))
                points = []
            image_draw = image_concat.copy()
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [0, 0, 255], -1) for punto in points1]
            [cv2.circle(image_draw, (punto[0] + img.shape[1], punto[1]), 3, [255, 0, 0], -1) for punto in points2]
    # Se necesitan minimo 4 puntos en cada imagen para poder continuar
    N = min(len(points1), len(points2))

    cv2.destroyAllWindows()
    assert N >= 4, 'At least four points are required'

    pts1 = np.array(points1[:N])
    pts2 = np.array(points2[:N])
    # Se encuentran los H
    if False:
        H, _ = cv2.findHomography(pts1, pts2, method=0)
    else:
        H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
    return H


# Funcion para realizar el promedio de las imagenes
def img_prom(img_1, img_2):
    # Se binarizan
    _, Ibw_1 = cv2.threshold(img_1[..., 0], 1, 255, cv2.THRESH_BINARY)
    _, Ibw_2 = cv2.threshold(img_2[..., 0], 1, 255, cv2.THRESH_BINARY)

    # Se realiza un bitwise and entre las dos imagenes binarizadas para obtener una mascara
    mask = cv2.bitwise_and(Ibw_1, Ibw_2)

    # Se realizo el merge con la mascara creada
    img_1_l = cv2.bitwise_and(img_1, cv2.merge((mask, mask, mask)))
    img_2_l = cv2.bitwise_and(img_2, cv2.merge((mask, mask, mask)))

    img_2_l = np.uint32(img_2_l)
    img_1_l = np.uint32(img_1_l)

    img = np.uint8((img_2_l + img_1_l) // 2)

    # Se crea una mascara negada
    n_mask = cv2.bitwise_not(mask)

    # Se realiza un merge con la nueva mascara
    img_1 = cv2.bitwise_and(img_1, cv2.merge((n_mask, n_mask, n_mask)))
    img_2 = cv2.bitwise_and(img_2, cv2.merge((n_mask, n_mask, n_mask)))

    # Se obtiene la imagen final
    img = cv2.bitwise_or(img, img_1)
    img = cv2.bitwise_or(img, img_2)
    return img


if __name__ == '__main__':
    # Se realiza la lectura de las imagenes
    N_images, img_list = img_lect()

    # El usuario ingresa la referencia
    referencia = int(input("Escoja la imagen de referencia, debe ser un numero entre 1 y " + str(N_images)))
    referencia -= 1

    # La referencia debe estar entre el numero de imagenes usadas
    assert N_images >= referencia > 0, "Error el numero ingresado debe ser entre 1 y N"

    H = []

    # Se realiza la homografia a cada una de las imagenes concatenadas y se guarda en la lista H
    for i in range(N_images - 1):
        a = homografia(img_list[i], img_list[(i + 1) % len(img_list)])
        H.append(a)

    factor = 10

    # matriz identidad
    I = np.identity(H[-1].shape[0])
    des = 2000
    h_traslacion = np.array([[1, 0, des], [0, 1, des], [0, 0, 1]], np.float64)

    img_transform = []
    img_recortada = []

    # For para tener en cuenta la referencia y tener finalmente la lista de las imagenes transformadas
    for i in range(N_images):
        h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
        print("Iter es:", i)
        if (i > referencia):
            for cont, j in enumerate(H[referencia:i]):
                h = j @ h
            h = np.linalg.inv(h)
        elif i < referencia:
            for j in (H[i:referencia]):
                h = h @ j
        if i != referencia:
            img_wrap = cv2.warpPerspective(img_list[i], h_traslacion @ h,
                                           (img_list[0].shape[1] * (factor), img_list[0].shape[0] * (factor)))

        else:
            img_wrap = cv2.warpPerspective(img_list[i], h_traslacion,
                                           (img_list[0].shape[1] * (factor), img_list[0].shape[0] * (factor)))

        img_transform.append(img_wrap)

    prom = np.zeros_like(img_transform[i])
    # For para realizar el promedio de las imagenes obtenidas previamente
    for idx, img in enumerate(img_transform):
        prom = img_prom(prom, img)
    cv2.imwrite("imagenfinal.png", prom)
    cv2.waitKey(0)
